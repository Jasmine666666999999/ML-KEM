# 该文件实现了针对 ML-KEM 的 'Roulette' 故障攻击，并包含了一个完整的攻击模拟流程。

import math
import time
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import binom, norm
from Crypto.Hash import SHA3_512, SHA3_256, SHAKE128, SHAKE256
from Crypto.Cipher import AES


# --- 随机数生成器 (RNG) ---

# RNG的抽象基类
class RNG(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def Bytes(self, nb_of_bytes):
        pass


# 使用操作系统随机源的RNG
class RNG_OS(RNG):
    def __init__(self):
        pass

    def Bytes(self, nb_of_bytes):
        return np.random.randint(
            0,
            high=256,
            size=nb_of_bytes,
            dtype=np.uint8)


# 使用AES-CTR实现的确定性RNG，用于测试
class RNG_AES_CTR(RNG):
    def __init__(self, seed):
        (K, V) = (bytes(32), bytes(16))
        self.cipher = AES.new(
            K,
            AES.MODE_CTR,
            nonce=b'',
            initial_value=V)
        self.cipher.encrypt(bytes(16))
        self.Update(seed=seed)

    def Bytes(self, nb_of_bytes):
        n = 16 * int(math.ceil(nb_of_bytes / 16))
        pt = bytes(n)
        ct = self.cipher.encrypt(pt)
        ct = np.frombuffer(ct[:nb_of_bytes], dtype=np.uint8)
        self.Update()
        return ct

    def Update(self, seed=None):
        if seed is None:
            pt = bytes(3 * 16)
        else:
            pt = seed.tobytes()
        ct = self.cipher.encrypt(pt)
        (K, V) = (ct[:32], ct[32:])
        self.cipher = AES.new(
            K,
            AES.MODE_CTR,
            nonce=b'',
            initial_value=V)
        self.cipher.encrypt(bytes(16))


# ML-KEM 的底层公钥加密（PKE）方案实现
class ML_KEM_PKE_Attack:
    N = 256
    Q = 3329
    QINV = np.int32(62209)
    SYMBYTES = 32
    ZETAS = np.array([
        2285, 2571, 2970, 1812, 1493, 1422, 287, 202,
        3158, 622, 1577, 182, 962, 2127, 1855, 1468,
        573, 2004, 264, 383, 2500, 1458, 1727, 3199,
        2648, 1017, 732, 608, 1787, 411, 3124, 1758,
        1223, 652, 2777, 1015, 2036, 1491, 3047, 1785,
        516, 3321, 3009, 2663, 1711, 2167, 126, 1469,
        2476, 3239, 3058, 830, 107, 1908, 3082, 2378,
        2931, 961, 1821, 2604, 448, 2264, 677, 2054,
        2226, 430, 555, 843, 2078, 871, 1550, 105,
        422, 587, 177, 3094, 3038, 2869, 1574, 1653,
        3083, 778, 1159, 3182, 2552, 1483, 2727, 1119,
        1739, 644, 2457, 349, 418, 329, 3173, 3254,
        817, 1097, 603, 610, 1322, 2044, 1864, 384,
        2114, 3193, 1218, 1994, 2455, 220, 2142, 1670,
        2144, 1799, 2051, 794, 1819, 2475, 2459, 478,
        3221, 3021, 996, 991, 958, 1869, 1522, 1628],
        dtype=np.int16)
    ZETASINV = np.array([
        1701, 1807, 1460, 2371, 2338, 2333, 308, 108,
        2851, 870, 854, 1510, 2535, 1278, 1530, 1185,
        1659, 1187, 3109, 874, 1335, 2111, 136, 1215,
        2945, 1465, 1285, 2007, 2719, 2726, 2232, 2512,
        75, 156, 3000, 2911, 2980, 872, 2685, 1590,
        2210, 602, 1846, 777, 147, 2170, 2551, 246,
        1676, 1755, 460, 291, 235, 3152, 2742, 2907,
        3224, 1779, 2458, 1251, 2486, 2774, 2899, 1103,
        1275, 2652, 1065, 2881, 725, 1508, 2368, 398,
        951, 247, 1421, 3222, 2499, 271, 90, 853,
        1860, 3203, 1162, 1618, 666, 320, 8, 2813,
        1544, 282, 1838, 1293, 2314, 552, 2677, 2106,
        1571, 205, 2918, 1542, 2721, 2597, 2312, 681,
        130, 1602, 1871, 829, 2946, 3065, 1325, 2756,
        1861, 1474, 1202, 2367, 3147, 1752, 2707, 171,
        3127, 3042, 1907, 1836, 1517, 359, 758, 1441],
        dtype=np.int16)
    MONTGOMERY_C = np.int16(np.uint64(0x100000000) % Q)

    # 初始化PKE方案参数
    def __init__(self, K, rng, seed=None):
        if not isinstance(K, int):
            raise TypeError("K 必须是整数")
        if K not in [2, 3, 4]:
            raise ValueError("K 必须是 2, 3 或 4")
        self.K = K
        self.ETA1 = 3 if self.K == 2 else 2
        self.ETA2 = 2
        self.DU = 11 if self.K == 4 else 10
        self.DV = 5 if self.K == 4 else 4
        self.rng = rng

    # PKE密钥生成
    def keygen(self, return_internals=False):
        seed = self.rng.Bytes(ML_KEM_PKE_Attack.SYMBYTES)
        (rho, sigma) = np.split(ML_KEM_PKE_Attack.G(seed), 2)
        A = self.generate_A(rho)
        s = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            data = ML_KEM_PKE_Attack.PRF(sigma, np.uint8(i), 64 * self.ETA1)
            s[i] = ML_KEM_PKE_Attack.CBD(data, self.ETA1)
        e = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            data = ML_KEM_PKE_Attack.PRF(sigma, np.uint8(self.K + i), 64 * self.ETA1)
            e[i] = ML_KEM_PKE_Attack.CBD(data, self.ETA1)
        s_ntt = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            s_ntt[i] = ML_KEM_PKE_Attack.NTT(s[i])
        t = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            t[i] = ML_KEM_PKE_Attack.Montgomery_dot(A[i, :, :], s_ntt)
            t[i] = ML_KEM_PKE_Attack.ToMontgomery(t[i])
            t[i] += ML_KEM_PKE_Attack.NTT(e[i])
        t %= ML_KEM_PKE_Attack.Q
        public_key = np.concatenate((ML_KEM_PKE_Attack.encode(t.ravel(), 12), rho))
        private_key = ML_KEM_PKE_Attack.encode(s_ntt.ravel(), 12)
        return [public_key, private_key, s, e] if return_internals else \
            [public_key, private_key]

    # PKE加密
    def encrypt(self, public_key, message, coins, return_internals=False,
                roulette_index=None):
        (t, rho) = np.split(public_key, [self.K * ML_KEM_PKE_Attack.N * 3 // 2])
        t = np.reshape(ML_KEM_PKE_Attack.decode(t, 12,
                                                self.K * ML_KEM_PKE_Attack.N), (self.K, -1))
        A_trans = self.generate_A(rho, transpose=True)
        r = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            data = ML_KEM_PKE_Attack.PRF(coins, np.uint8(i), 64 * self.ETA1)
            r[i] = ML_KEM_PKE_Attack.CBD(data, self.ETA1)
        e1 = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            data = ML_KEM_PKE_Attack.PRF(coins, np.uint8(self.K + i), 64 * self.ETA2)
            e1[i] = ML_KEM_PKE_Attack.CBD(data, self.ETA2)
        data = ML_KEM_PKE_Attack.PRF(coins, np.uint8(2 * self.K), 64 * self.ETA2)
        e2 = ML_KEM_PKE_Attack.CBD(data, self.ETA2)
        r_ntt = np.zeros((self.K, ML_KEM_PKE_Attack.N), dtype=np.int16)
        for i in range(self.K):
            r_ntt[i] = ML_KEM_PKE_Attack.NTT(r[i])
        u = np.copy(e1)
        for i in range(self.K):
            u[i] += ML_KEM_PKE_Attack.INTT(ML_KEM_PKE_Attack.Montgomery_dot(A_trans[i], r_ntt))
        v = ML_KEM_PKE_Attack.INTT(ML_KEM_PKE_Attack.Montgomery_dot(t, r_ntt))
        message = np.unpackbits(message, bitorder='little')
        v += e2 + ML_KEM_PKE_Attack.decompress(message, 1)
        (u, v) = (u % ML_KEM_PKE_Attack.Q, v % ML_KEM_PKE_Attack.Q)
        u2 = ML_KEM_PKE_Attack.compress(u, self.DU)
        v2 = ML_KEM_PKE_Attack.compress(v, self.DV)
        if return_internals:
            du = ML_KEM_PKE_Attack.mod_centered(ML_KEM_PKE_Attack.decompress(u2, self.DU) - u)
            dv = ML_KEM_PKE_Attack.mod_centered(ML_KEM_PKE_Attack.decompress(v2, self.DV) - v)
        if roulette_index is not None:
            v2[roulette_index] = (v2[roulette_index] + 2 ** (self.DV - 2)) \
                                 % (2 ** (self.DV))
        u = ML_KEM_PKE_Attack.encode(u2.ravel(), self.DU)
        v = ML_KEM_PKE_Attack.encode(v2, self.DV)
        c = np.concatenate((u, v))
        return [c, r, e1, du, e2, dv] if return_internals else [c]

    # PKE解密
    def decrypt(self, private_key, c):
        (u, v) = np.split(c, [self.K * ML_KEM_PKE_Attack.N * self.DU // 8])
        u = ML_KEM_PKE_Attack.decode(u, self.DU, self.K * ML_KEM_PKE_Attack.N)
        v = ML_KEM_PKE_Attack.decode(v, self.DV, ML_KEM_PKE_Attack.N)
        u = ML_KEM_PKE_Attack.decompress(u, self.DU)
        u = u.reshape((self.K, -1))
        v = ML_KEM_PKE_Attack.decompress(v, self.DV)
        for i in range(self.K):
            u[i] = ML_KEM_PKE_Attack.NTT(u[i])
        s = ML_KEM_PKE_Attack.decode(private_key, 12, self.K * ML_KEM_PKE_Attack.N)
        s = s.reshape((self.K, -1))
        m = v - ML_KEM_PKE_Attack.INTT(ML_KEM_PKE_Attack.Montgomery_dot(s, u))
        m = ML_KEM_PKE_Attack.compress(m % ML_KEM_PKE_Attack.Q, 1)
        m = np.packbits(m, bitorder='little')
        return m

    # --- 底层密码学辅助函数 ---

    def generate_A(self, rho, transpose=False):
        A = np.zeros((self.K, self.K, ML_KEM_PKE_Attack.N), dtype=np.uint16)
        for i in range(self.K):
            for j in range(self.K):
                ind = [i, j] if transpose else [j, i]
                seed = np.concatenate((rho, np.array(ind, dtype=np.uint8)))
                A[i, j, :] = ML_KEM_PKE_Attack.parse(ML_KEM_PKE_Attack.XOF(seed, ML_KEM_PKE_Attack.N * 3))
        return A

    @staticmethod
    def mod_centered(x):
        c = np.int16(ML_KEM_PKE_Attack.Q // 2)
        return ((x.astype(np.int16) + c) % ML_KEM_PKE_Attack.Q) - c

    @staticmethod
    def G(data):
        sha3 = SHA3_512.new()
        sha3.update(data.tobytes())
        return np.frombuffer(sha3.digest(), dtype=np.uint8)

    @staticmethod
    def XOF(data, nb_of_bytes):
        shake = SHAKE128.new()
        shake.update(data.tobytes())
        return np.frombuffer(shake.read(nb_of_bytes), dtype=np.uint8)

    @staticmethod
    def parse(data):
        L = len(data)
        d = ML_KEM_PKE_Attack.decode(data, 12, L * 2 // 3)
        d_is_valid = d < ML_KEM_PKE_Attack.Q
        ind = np.searchsorted(np.cumsum(d_is_valid), ML_KEM_PKE_Attack.N)
        if ind == L:
            raise Error("有效索引数量不足")
        d_is_valid[ind + 1:] = False
        return d[d_is_valid]

    @staticmethod
    def PRF(s, b, nb_of_bytes):
        shake = SHAKE256.new()
        shake.update(s.tobytes() + b.tobytes())
        return np.frombuffer(shake.read(nb_of_bytes), dtype=np.uint8)

    @staticmethod
    def CBD(data, eta):
        L = 64 * eta
        if len(data) != L:
            raise TypeError("无效长度")
        bits = np.unpackbits(data, bitorder='little')
        bits = np.reshape(bits, (ML_KEM_PKE_Attack.N, 2, eta))
        bits = np.sum(bits.astype(np.int8), axis=2)
        return bits[:, 0] - bits[:, 1]

    @staticmethod
    def CooleyTukeyButterflies(a, b, zeta):
        p = ML_KEM_PKE_Attack.Montgomery_multiply(b, zeta)
        return (a + p, a - p)

    @staticmethod
    def GentlemanSandeButterflies(a, b, zeta):
        return ((a + b) % ML_KEM_PKE_Attack.Q, ML_KEM_PKE_Attack.Montgomery_multiply(a - b, zeta))

    @staticmethod
    def NTT(data):
        data = data.astype(np.int16)
        L = [128, 64, 32, 16, 8, 4, 2]
        for i in range(7):
            ind_a = np.tile(np.arange(L[i]), 2 ** i) \
                    + np.repeat(np.arange(256, step=2 * L[i]), L[i])
            ind_b = ind_a + L[i]
            ind_z = np.repeat(np.arange(2 ** i, 2 ** (i + 1)), L[i])
            (data[ind_a], data[ind_b]) = ML_KEM_PKE_Attack.CooleyTukeyButterflies(
                data[ind_a], data[ind_b], ML_KEM_PKE_Attack.ZETAS[ind_z])
        return data % ML_KEM_PKE_Attack.Q

    @staticmethod
    def INTT(data):
        data = data.astype(np.int16)
        L = [128, 64, 32, 16, 8, 4, 2]
        j = 0
        for i in np.flip(np.arange(7)):
            ind_a = np.tile(np.arange(L[i]), 2 ** i) \
                    + np.repeat(np.arange(256, step=2 * L[i]), L[i])
            ind_b = ind_a + L[i]
            ind_z = np.repeat(np.arange(j, j + 2 ** i), L[i])
            (data[ind_a], data[ind_b]) = ML_KEM_PKE_Attack.GentlemanSandeButterflies(
                data[ind_a], data[ind_b], ML_KEM_PKE_Attack.ZETASINV[ind_z])
            j += 2 ** i
        return ML_KEM_PKE_Attack.Montgomery_multiply(data,
                                                     np.repeat(ML_KEM_PKE_Attack.ZETASINV[127], 256))

    @staticmethod
    def round_half_up(x):
        d = np.floor(x).astype(np.int16)
        return d + (x - d >= 0.5)

    @staticmethod
    def compress(x, delta):
        f = (2 ** delta) / ML_KEM_PKE_Attack.Q
        return ML_KEM_PKE_Attack.round_half_up(f * x) % (2 ** delta)

    @staticmethod
    def decompress(x, delta):
        f = ML_KEM_PKE_Attack.Q / (2 ** delta)
        return ML_KEM_PKE_Attack.round_half_up(f * x)

    @staticmethod
    def Montgomery_reduce(a):
        x = a.astype(np.int32)
        u = (x * ML_KEM_PKE_Attack.QINV).astype(np.int16)
        t = np.int32(u) * np.int32(ML_KEM_PKE_Attack.Q)
        t = x - t
        t >>= 16
        return t.astype(np.int16)

    @staticmethod
    def Montgomery_multiply(a, b):
        c = np.multiply(a.astype(np.int32), b.astype(np.int32))
        return ML_KEM_PKE_Attack.Montgomery_reduce(c)

    @staticmethod
    def ToMontgomery(a):
        return ML_KEM_PKE_Attack.Montgomery_multiply(a, ML_KEM_PKE_Attack.MONTGOMERY_C)

    @staticmethod
    def Montgomery_basecase_multiply(a, b):
        r = np.zeros((256), dtype=np.int16)
        r[0::2] = ML_KEM_PKE_Attack.Montgomery_multiply(a[1::2], b[1::2])
        r[0::4] = ML_KEM_PKE_Attack.Montgomery_multiply(r[0::4], ML_KEM_PKE_Attack.ZETAS[64:])
        r[2::4] = ML_KEM_PKE_Attack.Montgomery_multiply(r[2::4], -ML_KEM_PKE_Attack.ZETAS[64:])
        r[0::2] += ML_KEM_PKE_Attack.Montgomery_multiply(a[0::2], b[0::2])
        r[1::2] = ML_KEM_PKE_Attack.Montgomery_multiply(a[0::2], b[1::2])
        r[1::2] += ML_KEM_PKE_Attack.Montgomery_multiply(a[1::2], b[0::2])
        return r % ML_KEM_PKE_Attack.Q

    @staticmethod
    def Montgomery_dot(a, b):
        c = ML_KEM_PKE_Attack.Montgomery_basecase_multiply(a[0], b[0])
        for i in range(1, a.shape[0]):
            c += ML_KEM_PKE_Attack.Montgomery_basecase_multiply(a[i], b[i])
        return c % ML_KEM_PKE_Attack.Q

    @staticmethod
    def encode(a, nb_of_bits):
        r = np.zeros(len(a) * 2, dtype=np.uint8)
        r[0::2] = np.bitwise_and(a, 0xff).astype(np.uint8)
        r[1::2] = np.bitwise_and(a >> 8, 0xff).astype(np.uint8)
        r = np.unpackbits(r, bitorder='little')
        r = r.reshape((-1, 16))
        r = r[:, :nb_of_bits].ravel()
        return np.packbits(r, bitorder='little')

    @staticmethod
    def decode(a, nb_of_bits, nb_of_symbols):
        r = np.unpackbits(a, bitorder='little')
        r = r[:nb_of_symbols * nb_of_bits]
        r = r.reshape((nb_of_symbols, nb_of_bits))
        r = np.pad(r, ((0, 0), (0, 16 - nb_of_bits)),
                   mode='constant', constant_values=(0))
        r = np.packbits(r.ravel(), bitorder='little')
        r = r.astype(np.uint16)
        return np.bitwise_or(r[0::2], r[1::2] << 8)


# ML-KEM 的密钥封装机制（KEM）实现
class ML_KEM_KEM_Attack:
    SHARED_SECRET_BYTES = 32

    # 初始化KEM
    def __init__(self, K, rng):
        self.pke = ML_KEM_PKE_Attack(K, rng)

    # KEM密钥对生成
    def keygen(self, return_internals=False):
        public_key, private_key, *internals = self.pke.keygen(
            return_internals=return_internals)
        h = ML_KEM_KEM_Attack.H(public_key)
        z = self.pke.rng.Bytes(ML_KEM_KEM_Attack.SHARED_SECRET_BYTES)
        private_key = np.concatenate((private_key, public_key, h, z))
        return public_key, private_key, *internals

    # KEM密钥封装
    def encapsulate(self, public_key, return_internals=False):
        m = self.pke.rng.Bytes(ML_KEM_PKE_Attack.SYMBYTES)
        m = ML_KEM_KEM_Attack.H(m)
        h = ML_KEM_KEM_Attack.H(public_key)
        (k, r) = np.split(ML_KEM_PKE_Attack.G(np.concatenate((m, h))), 2)
        c, *internals = self.pke.encrypt(public_key, m, r,
                                         return_internals=return_internals)
        k2 = np.concatenate((k, ML_KEM_KEM_Attack.H(c)))
        k2 = ML_KEM_KEM_Attack.KDF(k2)
        return [c, k2, m, *internals, k] if return_internals else [c, k2]

    # KEM密钥解封装
    def decapsulate(self, private_key, c, return_internals=False,
                    roulette_index=None):
        l = 12 * self.pke.K * ML_KEM_PKE_Attack.N // 8
        sk, pk, h, z = np.split(private_key, [l, 2 * l + 32, 2 * l + 64])
        m = self.pke.decrypt(sk, c)
        (k, r) = np.split(ML_KEM_PKE_Attack.G(np.concatenate((m, h))), 2)
        [c2] = self.pke.encrypt(pk, m, r, roulette_index=roulette_index)
        if not np.array_equal(c, c2):
            k = z
        k = np.concatenate((k, ML_KEM_KEM_Attack.H(c)))
        k = ML_KEM_KEM_Attack.KDF(k)
        return [k, m] if return_internals else k

    # 获取当前参数集版本名称
    def version(self):
        if self.pke.K == 2:
            return "ML-KEM-512"
        elif self.pke.K == 3:
            return "ML-KEM-768"
        else:
            return "ML-KEM-1024"

    @staticmethod
    def H(data):
        sha3 = SHA3_256.new()
        sha3.update(data.tobytes())
        d = sha3.digest()
        return np.frombuffer(d, dtype=np.uint8)

    @staticmethod
    def KDF(x):
        shake = SHAKE256.new()
        shake.update(x.tobytes())
        k = shake.read(ML_KEM_KEM_Attack.SHARED_SECRET_BYTES)
        return np.frombuffer(k, dtype=np.uint8)


# --- 攻击核心逻辑函数 ---

# 将多项式转换为用于矩阵乘法的特定矩阵
def poly_multiplier_to_matrix(poly, row_index=None):
    p = poly.astype(np.int16)
    if row_index is None:
        N = len(p)
        r = np.zeros((N, N), dtype=np.int16)
        for i in range(N):
            r[i, :] = np.concatenate((np.flip(p[:i + 1]), -np.flip(p[i + 1:])))
    else:
        r = np.concatenate((np.flip(p[:row_index + 1]),
                            -np.flip(p[row_index + 1:])))
    return r


# 根据'Roulette'攻击的原理篡改密文
def manipulate_ciphertext(pke, c, index):
    c2 = np.copy(c)
    nb_of_bytes = ML_KEM_PKE_Attack.N * pke.DV // 8
    v = ML_KEM_PKE_Attack.decode(c[-nb_of_bytes:], pke.DV, ML_KEM_PKE_Attack.N)
    v_mod = (v[index] + 2 ** (pke.DV - 2)) % (2 ** (pke.DV))
    error = ML_KEM_PKE_Attack.decompress(v_mod, pke.DV) \
            - ML_KEM_PKE_Attack.decompress(v[index], pke.DV)
    error %= ML_KEM_PKE_Attack.Q
    v[index] = v_mod
    v = ML_KEM_PKE_Attack.encode(v, pke.DV)
    c2[-nb_of_bytes:] = v
    return [c2, error]


# 通过多次封装和篡改，生成用于求解私钥的线性不等式系统
def generate_inequalities(
        kem,
        public_key,
        nb_of_inequalities,
        index=None,
        bias_threshold=None,
        max_nb_of_encapsulations=None,
        return_manipulation=False,
        verbose=True):
    if verbose:
        print("正在为 {:s} 生成 {:d} 个不等式..."
              .format(kem.version(), nb_of_inequalities))
    (K, N) = (kem.pke.K, ML_KEM_PKE_Attack.N)
    a = np.zeros((nb_of_inequalities, 2 * K * N), dtype=np.int16)
    b = np.zeros((nb_of_inequalities), dtype=np.int16)
    if return_manipulation:
        manipulated_indices = np.zeros(nb_of_inequalities, dtype=np.uint8)
        ciphertexts = np.zeros((nb_of_inequalities,
                                (K * kem.pke.DU + kem.pke.DV) * N // 8), dtype=np.uint8)
        manipulated_ciphertexts = np.zeros((nb_of_inequalities,
                                            (K * kem.pke.DU + kem.pke.DV) * N // 8), dtype=np.uint8)
        shared_secrets = np.zeros((nb_of_inequalities,
                                   ML_KEM_KEM_Attack.SHARED_SECRET_BYTES), dtype=np.uint8)
        manipulated_shared_secrets = np.zeros((nb_of_inequalities,
                                               ML_KEM_KEM_Attack.SHARED_SECRET_BYTES), dtype=np.uint8)
    for i in range(nb_of_inequalities):
        done = False
        nb_of_encapsulations = 0
        lowest_bias = 10000
        while not done:
            c, ss, m, r, e1, du, e2, dv, ss_pre = kem.encapsulate(public_key,
                                                                  return_internals=True)
            b2 = e2 + dv
            m = np.unpackbits(m, bitorder='little')
            _, error = manipulate_ciphertext(kem.pke, c, np.arange(N))
            b2[np.logical_and(error == 832, m == 0)] -= 1
            b2[np.logical_and(error == 833, m == 1)] += 1
            bias = abs(b2)
            ind = np.argmin(bias) if index is None else index
            if return_manipulation:
                manipulated_indices[i] = ind
                ciphertexts[i, :] = c
                manipulated_ciphertexts[i, :], _ = \
                    manipulate_ciphertext(kem.pke, c, ind)
                shared_secrets[i, :] = ss
                manipulated_shared_secrets[i, :] = ML_KEM_KEM_Attack.KDF(
                    np.concatenate((ss_pre,
                                    ML_KEM_KEM_Attack.H(manipulated_ciphertexts[i, :]))))
            if bias[ind] < lowest_bias:
                lowest_bias = bias[ind]
                b[i] = b2[ind]
                for j in range(K):
                    a[i, j * N:(j + 1) * N] = poly_multiplier_to_matrix(
                        -e1[j, :] - du[j, :], row_index=ind)
                    a[i, (K + j) * N:(K + j + 1) * N] = poly_multiplier_to_matrix(
                        r[j, :], row_index=ind)
            nb_of_encapsulations += 1
            done = (bias_threshold is None) \
                   or (bias[ind] <= bias_threshold) \
                   or ((max_nb_of_encapsulations is not None) and \
                       (nb_of_encapsulations >= max_nb_of_encapsulations))
    return [a, b, manipulated_indices, ciphertexts, manipulated_ciphertexts, \
            shared_secrets, manipulated_shared_secrets] if return_manipulation \
        else [a, b]


# 通过实际解封装来判断不等式结果（慢速，用于验证）
def evaluate_inequalities_slow(
        kem,
        private_key,
        manipulated_indices,
        manipulated_ciphertexts,
        manipulated_shared_secrets):
    nb_of_inequalities = manipulated_ciphertexts.shape[0]
    is_geq_zero = np.full((nb_of_inequalities), False)
    for i in range(nb_of_inequalities):
        ss = kem.decapsulate(private_key, manipulated_ciphertexts[i, :],
                             roulette_index=manipulated_indices[i])
        is_geq_zero[i] = not np.array_equal(ss,
                                            manipulated_shared_secrets[i, :])
    return is_geq_zero


# 利用已知私钥直接计算不等式结果（快速，用于模拟）
def evaluate_inequalities_fast(a, b, solution):
    return (np.matmul(a, solution) + b) >= 0


# 模拟不等式在生成过程中可能出现的错误
def corrupt_inequalities(is_geq_zero, prob_success_is_missed, verbose=True):
    is_geq_zero_corrupt = np.copy(is_geq_zero)
    ind, = np.where(is_geq_zero == False)
    miss = np.random.binomial(1, prob_success_is_missed, size=len(ind))
    ind2, = np.where(miss == 1)
    is_geq_zero_corrupt[ind[ind2]] = True
    print("已损坏 {:d} 个不等式中的 {:d} 个"
          .format(len(ind2), len(is_geq_zero)))
    return is_geq_zero_corrupt


# 生成关于私钥的等式关系，辅助求解
def generate_equalities(kem, public_key, verbose=True):
    if verbose:
        print("正在为 {:s} 生成等式...".format(kem.version()))
    (K, N) = (kem.pke.K, ML_KEM_PKE_Attack.N)
    (t, rho) = np.split(public_key, [K * N * 3 // 2])
    t = np.reshape(ML_KEM_PKE_Attack.decode(t, 12, K * N), (K, N))
    A = kem.pke.generate_A(rho, transpose=False)
    a = np.hstack((np.zeros((K * N, K * N), dtype=np.int32),
                   2285 * np.eye(K * N, dtype=np.int32)))
    b = np.zeros((K * N), dtype=np.int32)
    for i in range(K):
        b[i * N:(i + 1) * N] = -kem.pke.INTT(t[i])
        for j in range(K):
            a[i * N:(i + 1) * N, j * N:(j + 1) * N] = poly_multiplier_to_matrix(
                kem.pke.INTT(A[i, j]))
    return a, b


# 核心求解器：使用置信度传播算法求解不等式系统，恢复私钥
def solve_inequalities(kem, a, b, is_geq_zero,
                       max_nb_of_iterations=16,
                       verbose=True,
                       solution=None):
    if verbose:
        print("正在求解不等式...")
    eta = kem.pke.ETA1
    [nb_of_inequalities, nb_of_unknowns] = a.shape
    guess = np.zeros((nb_of_unknowns), dtype=int)
    if verbose and solution is not None:
        nb_correct = np.count_nonzero(solution == guess)
        print("正确猜出的未知数数量: {:d}/{:d}"
              .format(nb_correct, len(solution)))
    if nb_of_inequalities == 0:
        return guess
    nb_of_values = 2 * eta + 1
    x = np.arange(-eta, eta + 1, dtype=np.int8)
    x_pmf = binom.pmf(x + eta, 2 * eta, 0.5)
    x_pmf = np.repeat(x_pmf.reshape(1, -1), nb_of_unknowns, axis=0)
    a = a.astype(np.int16)
    a_squared = np.square(a)
    prob_geq_zero = np.zeros((nb_of_inequalities), dtype=float)
    p_failure_is_observed = np.count_nonzero(is_geq_zero) / nb_of_inequalities
    mean = np.matmul(x_pmf, x)
    variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
    mean = np.matmul(a, mean)
    variance = np.matmul(a_squared, variance)
    zscore = np.divide(mean + 0.5 + b, np.sqrt(variance))
    p_failure_is_reality = norm.cdf(zscore)
    p_failure_is_reality = np.mean(p_failure_is_reality)
    p_inequality_is_correct = min(
        p_failure_is_reality / p_failure_is_observed, 1.0)
    prob_geq_zero[is_geq_zero] = p_inequality_is_correct
    fitness = np.zeros((max_nb_of_iterations), dtype=float)
    fitness_max = np.sum(np.maximum(prob_geq_zero, 1 - prob_geq_zero))
    for z in range(max_nb_of_iterations):
        if verbose:
            print("迭代 " + str(z))
            time_start = time.time()
        mean = np.matmul(x_pmf, x)
        variance = np.matmul(x_pmf, np.square(x)) - np.square(mean)
        mean = np.multiply(a, np.repeat(mean[np.newaxis, :],
                                        nb_of_inequalities, axis=0))
        variance = np.multiply(
            a_squared,
            np.repeat(variance[np.newaxis, :], nb_of_inequalities, axis=0))
        mean = mean.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns, axis=1) \
               - mean
        mean += b[:, np.newaxis]
        variance = variance.sum(axis=1).reshape(-1, 1).repeat(nb_of_unknowns,
                                                              axis=1) - variance
        variance = np.clip(variance, 1, None)
        psuccess = np.zeros((nb_of_values, nb_of_inequalities,
                             nb_of_unknowns), dtype=float)
        for j in range(nb_of_values):
            zscore = np.divide(a * x[j] + mean + 0.5, np.sqrt(variance))
            psuccess[j, :, :] = norm.cdf(zscore)
        psuccess = np.transpose(psuccess, axes=[2, 0, 1])
        psuccess = \
            np.multiply(psuccess, prob_geq_zero[np.newaxis, np.newaxis, :]) + \
            np.multiply(1 - psuccess, 1 - prob_geq_zero[np.newaxis, np.newaxis, :])
        psuccess = np.clip(psuccess, 10e-5, None)
        psuccess = np.sum(np.log(psuccess), axis=2)
        row_means = psuccess.max(axis=1)
        psuccess -= row_means[:, np.newaxis]
        psuccess = np.exp(psuccess)
        x_pmf = np.multiply(psuccess, x_pmf)
        row_sums = x_pmf.sum(axis=1)
        x_pmf /= row_sums[:, np.newaxis]
        guess = x[np.argmax(x_pmf, axis=1)]
        fit = (np.matmul(a, guess) + b >= 0).astype(float)
        fit = np.dot(fit, prob_geq_zero) + np.dot(1 - fit, 1 - prob_geq_zero)
        fitness[z] = fit / fitness_max
        if verbose:
            time_end = time.time()
            print("用时: {:.1f} 秒".format(time_end - time_start))
            print("适应度 {:.2f}%".format(fitness[z] * 100))
            if solution is not None:
                nb_correct = np.count_nonzero(solution == guess)
                print("正确猜出的未知数数量: {:d}/{:d}"
                      .format(nb_correct, len(solution)))
        if (z > 1) and fitness[z - 1] >= fitness[z]:
            break
    return guess


# --- 用于验证算法正确性的自检函数 ---
def test_inequalities():
    rng = RNG_OS()
    nb_of_inequalities = 1000
    for K in [2, 3, 4]:
        kem = ML_KEM_KEM_Attack(K, rng)
        [public_key, private_key, s, e] = kem.keygen(return_internals=True)
        solution = np.concatenate((s.ravel(), e.ravel()))
        [a, b, manipulated_indices, _, manipulated_ciphertexts, _,
         manipulated_shared_secrets] = generate_inequalities(kem,
                                                             public_key, nb_of_inequalities, return_manipulation=True)
        is_geq_zero1 = evaluate_inequalities_slow(kem, private_key,
                                                  manipulated_indices, manipulated_ciphertexts,
                                                  manipulated_shared_secrets)
        is_geq_zero2 = evaluate_inequalities_fast(a, b, solution)
        if np.any(is_geq_zero1 != is_geq_zero2):
            raise ValueError("不等式测试失败")
        print("不等式测试通过")


def test_equalities():
    rng = RNG_OS()
    for K in [2, 3, 4]:
        kem = ML_KEM_KEM_Attack(K, rng)
        [public_key, private_key, s, e] = kem.keygen(return_internals=True)
        solution = np.concatenate((s.ravel(), e.ravel()))
        [a, b] = generate_equalities(kem, public_key)
        r = (np.matmul(a, solution) + b) % ML_KEM_PKE_Attack.Q
        if r.any():
            raise ValueError("等式测试失败")
        print("等式测试通过")


def test():
    print("正在测试求解器...")
    test_inequalities()
    test_equalities()


# 主函数，执行完整的密钥恢复攻击流程
def main():
    # 运行自检
    print("正在运行求解器自检...")
    test_start = time.time()
    test()
    test_end = time.time()
    print(f"自检在 {test_end - test_start:.2f} 秒内完成\n")

    # 生成目标密钥对
    K = 3
    rng = RNG_OS()
    kem = ML_KEM_KEM_Attack(K, rng)

    print("正在生成密钥对...")
    key_start = time.time()
    public_key, private_key, s, e = kem.keygen(return_internals=True)
    key_end = time.time()
    secret = np.concatenate((s.ravel(), e.ravel()))
    secret_len = len(secret)
    print(f"密钥对生成在 {key_end - key_start:.2f} 秒内完成\n")

    # 循环尝试，逐步增加不等式数量进行攻击
    start_ineq = 1000
    step = 1000
    max_ineq = 10000

    total_start = time.time()
    for n_ineq in range(start_ineq, max_ineq + 1, step):
        print(f"正在尝试 {n_ineq} 个不等式...")
        iter_start = time.time()

        a, b = generate_inequalities(kem, public_key, n_ineq)
        signs = evaluate_inequalities_fast(a, b, secret)
        recovered = solve_inequalities(kem, a, b, signs, verbose=False)

        correct = np.count_nonzero(recovered == secret)
        perc = correct / secret_len * 100
        iter_end = time.time()

        print(f"已恢复 {correct}/{secret_len} 个系数 ({perc:.2f}%)")
        print(f"迭代用时: {iter_end - iter_start:.2f} 秒\n")

        if correct == secret_len:
            total_end = time.time()
            print(f"成功使用 {n_ineq} 个不等式恢复密钥！")
            print(f"总用时: {total_end - total_start:.2f} 秒")
            return

    total_end = time.time()
    print(f"尝试 {max_ineq} 个不等式后密钥恢复失败。")
    print(f"总用时: {total_end - total_start:.2f} 秒")


if __name__ == "__main__":
    main()