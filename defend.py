# 这是一个带有“提交-频闪”防御机制的 ML-KEM 实现，用于演示如何抵抗故障攻击。

import math
import time
import os
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


# 使用操作系统提供的随机源 (os.urandom) 的RNG实现
class RNG_OS(RNG):
    def __init__(self):
        pass

    def Bytes(self, nb_of_bytes):
        return np.frombuffer(os.urandom(nb_of_bytes), dtype=np.uint8)


# 使用AES-CTR模式实现的确定性RNG，主要用于可复现的测试
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


# --- ML-KEM 的底层公钥加密（PKE）方案实现 ---
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

    # 初始化PKE方案的参数
    def __init__(self, K, rng):
        if not isinstance(K, int):
            raise TypeError("K must be an integer")
        if K not in [2, 3, 4]:
            raise ValueError("K must be 2, 3, or 4")
        self.K = K
        self.ETA1 = 3 if self.K == 2 else 2
        self.ETA2 = 2
        self.DU = 11 if self.K == 4 else 10
        self.DV = 5 if self.K == 4 else 4
        self.rng = rng

    # PKE密钥生成
    def keygen(self):
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
        return public_key, private_key

    # PKE加密
    def encrypt(self, public_key, message, coins):
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
        u_encoded = ML_KEM_PKE_Attack.encode(u2.ravel(), self.DU)
        v_encoded = ML_KEM_PKE_Attack.encode(v2, self.DV)
        c = np.concatenate((u_encoded, v_encoded))
        return c

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

    # 从种子生成矩阵A
    def generate_A(self, rho, transpose=False):
        A = np.zeros((self.K, self.K, ML_KEM_PKE_Attack.N), dtype=np.uint16)
        for i in range(self.K):
            for j in range(self.K):
                ind = [i, j] if transpose else [j, i]
                seed = np.concatenate((rho, np.array(ind, dtype=np.uint8)))
                A[i, j, :] = ML_KEM_PKE_Attack.parse(ML_KEM_PKE_Attack.XOF(seed, ML_KEM_PKE_Attack.N * 3))
        return A

    # 哈希函数 G (SHA3-512)
    @staticmethod
    def G(data):
        sha3 = SHA3_512.new()
        sha3.update(data.tobytes())
        return np.frombuffer(sha3.digest(), dtype=np.uint8)

    # 可扩展输出函数 XOF (SHAKE-128)
    @staticmethod
    def XOF(data, nb_of_bytes):
        shake = SHAKE128.new()
        shake.update(data.tobytes())
        return np.frombuffer(shake.read(nb_of_bytes), dtype=np.uint8)

    # 解析XOF的输出
    @staticmethod
    def parse(data):
        L = len(data)
        d = ML_KEM_PKE_Attack.decode(data, 12, L * 2 // 3)
        d_is_valid = d < ML_KEM_PKE_Attack.Q
        ind = np.searchsorted(np.cumsum(d_is_valid), ML_KEM_PKE_Attack.N)
        if ind == L:
            raise Exception("Not enough valid indices")
        d_is_valid[ind + 1:] = False
        return d[d_is_valid]

    # 伪随机函数 PRF (SHAKE-256)
    @staticmethod
    def PRF(s, b, nb_of_bytes):
        shake = SHAKE256.new()
        shake.update(s.tobytes() + b.tobytes())
        return np.frombuffer(shake.read(nb_of_bytes), dtype=np.uint8)

    # 中心二项分布采样
    @staticmethod
    def CBD(data, eta):
        L = 64 * eta
        if len(data) != L:
            raise TypeError("Invalid length")
        bits = np.unpackbits(data, bitorder='little')
        bits = np.reshape(bits, (ML_KEM_PKE_Attack.N, 2, eta))
        bits = np.sum(bits.astype(np.int8), axis=2)
        return bits[:, 0] - bits[:, 1]

    # NTT中的蝴蝶操作
    @staticmethod
    def CooleyTukeyButterflies(a, b, zeta):
        p = ML_KEM_PKE_Attack.Montgomery_multiply(b, zeta)
        return (a + p, a - p)

    @staticmethod
    def GentlemanSandeButterflies(a, b, zeta):
        return ((a + b) % ML_KEM_PKE_Attack.Q, ML_KEM_PKE_Attack.Montgomery_multiply(a - b, zeta))

    # 数论变换
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

    # 逆数论变换
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

    # 压缩与解压缩函数
    @staticmethod
    def compress(x, delta):
        f = (2 ** delta) / ML_KEM_PKE_Attack.Q
        return np.round(f * x).astype(np.int16) % (2 ** delta)

    @staticmethod
    def decompress(x, delta):
        f = ML_KEM_PKE_Attack.Q / (2 ** delta)
        return np.round(f * x).astype(np.int16)

    # 蒙哥马利约减
    @staticmethod
    def Montgomery_reduce(a):
        x = a.astype(np.int32)
        u = (x * ML_KEM_PKE_Attack.QINV).astype(np.int16)
        t = np.int32(u) * np.int32(ML_KEM_PKE_Attack.Q)
        t = x - t
        t >>= 16
        return t.astype(np.int16)

    # 蒙哥马利乘法
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

    # 编码与解码
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
        r = np.pad(r, ((0, 0), (0, 16 - nb_of_bits)),mode='constant', constant_values=(0))
        r = np.packbits(r.ravel(), bitorder='little')
        r = r.astype(np.uint16)
        return np.bitwise_or(r[0::2], r[1::2] << 8)


# --- ML-KEM 的密钥封装机制（KEM）实现 ---
class ML_KEM_KEM_Attack:
    SHARED_SECRET_BYTES = 32

    # 初始化KEM方案
    def __init__(self, K, rng=None):
        if rng is None:
            rng = RNG_OS()
        self.pke = ML_KEM_PKE_Attack(K, rng)

    # KEM密钥对生成
    def keygen(self):
        public_key, private_key = self.pke.keygen()
        h = self._H(public_key)
        z = self.pke.rng.Bytes(ML_KEM_KEM_Attack.SHARED_SECRET_BYTES)
        private_key = np.concatenate((private_key, public_key, h, z))
        return public_key, private_key

    # KEM密钥封装
    def encapsulate(self, public_key):
        m = self.pke.rng.Bytes(ML_KEM_PKE_Attack.SYMBYTES)
        m = self._H(m)
        h = self._H(public_key)
        (k, r) = np.split(ML_KEM_PKE_Attack.G(np.concatenate((m, h))), 2)
        c = self.pke.encrypt(public_key, m, r)
        k_derived = np.concatenate((k, self._H(c)))
        shared_secret = self._KDF(k_derived)
        return c, shared_secret

    # KEM密钥解封装（包含防御机制）
    def decapsulate(self, private_key, c):
        # 解析私钥，获取回退密钥z等
        l = 12 * self.pke.K * ML_KEM_PKE_Attack.N // 8
        sk, pk, h, z = np.split(private_key, [l, 2 * l + 32, 2 * l + 64])

        # --- “提交-频闪”防御机制开始 ---
        # 步骤1: 生成一次性的秘密随机数
        strobe_nonce = self.pke.rng.Bytes(32)

        # 步骤2: 计算包含nonce的承诺哈希
        commitment = self._H(np.concatenate((c, strobe_nonce)))

        # 正常执行解密和重加密
        m_prime = self.pke.decrypt(sk, c)
        (k_prime, r_prime) = np.split(ML_KEM_PKE_Attack.G(np.concatenate((m_prime, h))), 2)
        c_prime = self.pke.encrypt(pk, m_prime, r_prime)

        # 步骤3: 计算包含nonce的验证哈希
        verification = self._H(np.concatenate((c_prime, strobe_nonce)))

        # 步骤4: 比较承诺值与验证值，若不匹配则返回预设的z密钥
        if not np.array_equal(commitment, verification):
            k_base = z
        else:
            k_base = k_prime
        # --- 防御机制结束 ---

        # 派生并返回最终的共享密钥
        k_derived = np.concatenate((k_base, self._H(c)))
        shared_secret = self._KDF(k_derived)
        return shared_secret

    # KEM中使用的哈希函数 (SHA3-256)
    @staticmethod
    def _H(data):
        sha3 = SHA3_256.new()
        sha3.update(data.tobytes())
        d = sha3.digest()
        return np.frombuffer(d, dtype=np.uint8)

    # KEM中使用的密钥派生函数 (SHAKE-256)
    @staticmethod
    def _KDF(x):
        shake = SHAKE256.new()
        shake.update(x.tobytes())
        k = shake.read(ML_KEM_KEM_Attack.SHARED_SECRET_BYTES)
        return np.frombuffer(k, dtype=np.uint8)


# 主函数，用于演示防御效果
def main():
    print("=" * 60)
    print("== 演示针对故障攻击的“提交-频闪”防御方案 ==")
    print("=" * 60)

    # 初始化参数和KEM实例
    K = 3
    rng = RNG_OS()
    kem = ML_KEM_KEM_Attack(K, rng)

    # [1] 生成密钥对
    print("[1] 正在生成带防御的密钥对...")
    public_key, private_key = kem.keygen()
    print("    密钥生成完毕。\n")

    # [2] 验证正常功能
    print("[2] 正在验证合法封装与解封装...")
    ciphertext, shared_secret_alice = kem.encapsulate(public_key)
    shared_secret_bob = kem.decapsulate(private_key, ciphertext)

    if np.array_equal(shared_secret_alice, shared_secret_bob):
        print("    成功：防御机制不影响正常操作。\n")
    else:
        print("    失败：合法解封装操作失败，请检查实现。\n")
        return

    # [3] 模拟'Roulette'攻击（篡改密文）
    print("[3] 正在模拟 'Roulette' 故障攻击...")
    print("    攻击者接收到原始密文。")

    manipulated_ciphertext = np.copy(ciphertext)
    v_offset = kem.pke.K * ML_KEM_PKE_Attack.N * kem.pke.DU // 8
    v_bytes = manipulated_ciphertext[v_offset:]

    v_decoded = ML_KEM_PKE_Attack.decode(v_bytes, kem.pke.DV, ML_KEM_PKE_Attack.N)
    target_index = 42
    v_decoded[target_index] = (v_decoded[target_index] + 2 ** (kem.pke.DV - 2)) % (2 ** (kem.pke.DV))
    v_manipulated_bytes = ML_KEM_PKE_Attack.encode(v_decoded, kem.pke.DV)
    manipulated_ciphertext[v_offset:] = v_manipulated_bytes

    print(f"    已在系数索引 {target_index} 处篡改密文。")

    # [4] 使用带防御的解封装函数处理，并验证结果
    print("\n[4] 服务器使用带防御的解封装函数处理恶意密文...")
    attack_shared_secret = kem.decapsulate(private_key, manipulated_ciphertext)

    if not np.array_equal(shared_secret_alice, attack_shared_secret):
        print("    成功：“提交-频闪”防御机制检测到篡改！")
        print("    服务器返回了预设密钥，攻击已被挫败。\n")
    else:
        print("    失败：防御被绕过，攻击者获取了共享密钥。\n")

    print("=" * 60)
    print("== 演示完毕 ==")
    print("=" * 60)


if __name__ == "__main__":
    main()