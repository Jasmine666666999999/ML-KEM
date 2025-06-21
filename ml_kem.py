# 该文件是 ML-KEM (Kyber) 算法的完整实现。
# 它包含了符合 FIPS 203 标准的所有核心密码学操作。

from test import test_all
from Crypto.Hash import SHAKE128, SHAKE256, SHA3_256, SHA3_512

# ML-KEM 不同安全级别的核心参数定义
ML_KEM_SPECIFICATIONS = {
    "ML-KEM-512": (2, 3, 2, 10, 4),
    "ML-KEM-768": (3, 2, 2, 10, 4),
    "ML-KEM-1024": (4, 2, 2, 11, 5)
}

# NTT（数论变换）所需的旋转因子（Zetas）
NTT_ZETAS = [
    1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797,
    2786, 3260, 569, 1746, 296, 2447, 1339, 1476, 3046, 56, 2240, 1333,
    1426, 2094, 535, 2882, 2393, 2879, 1974, 821, 289, 331, 3253, 1756,
    1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915,
    2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648,
    2474, 3110, 1227, 910, 17, 2761, 583, 2649, 1637, 723, 2288, 1100,
    1409, 2662, 3281, 233, 756, 2156, 3015, 3050, 1703, 1651, 2789, 1789,
    1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641,
    1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757,
    2099, 561, 2466, 2594, 2804, 1092, 403, 1026, 1143, 2150, 2775, 886,
    1722, 1212, 1874, 1029, 2110, 2935, 885, 2154
]

# NTT 点对乘法中使用的旋转因子
NTT_ZETAS_MUL = [
    17, -17, 2761, -2761, 583, -583, 2649, -2649, 1637, -1637, 723, -723,
    2288, -2288, 1100, -1100, 1409, -1409, 2662, -2662, 3281, -3281,
    233, -233, 756, -756, 2156, -2156, 3015, -3015, 3050, -3050, 1703, -1703,
    1651, -1651, 2789, -2789, 1789, -1789, 1847, -1847, 952, -952,
    1461, -1461, 2687, -2687, 939, -939, 2308, -2308, 2437, -2437,
    2388, -2388, 733, -733, 2337, -2337, 268, -268, 641, -641, 1584, -1584,
    2298, -2298, 2037, -2037, 3220, -3220, 375, -375, 2549, -2549,
    2090, -2090, 1645, -1645, 1063, -1063, 319, -319, 2773, -2773,
    757, -757, 2099, -2099, 561, -561, 2466, -2466, 2594, -2594,
    2804, -2804, 1092, -1092, 403, -403, 1026, -1026, 1143, -1143,
    2150, -2150, 2775, -2775, 886, -886, 1722, -1722, 1212, -1212,
    1874, -1874, 1029, -1029, 2110, -2110, 2935, -2935, 885, -885, 2154, -2154
]


# ML-KEM 密钥封装机制的完整实现
class ML_KEM:
    # 初始化 KEM 实例，设置对应安全级别的参数
    def __init__(self, spec_name='ML-KEM-768'):
        if spec_name not in ML_KEM_SPECIFICATIONS:
            raise ValueError("Unsupported ML-KEM specification")
        self.q = 3329
        self.n = 256
        self.k, self.eta1, self.eta2, self.du, self.dv = ML_KEM_SPECIFICATIONS[spec_name]
        self.spec_name = spec_name

    # 辅助函数：动态更新实例的安全级别
    def _update_spec(self, spec_name):
        if spec_name and self.spec_name != spec_name:
            self.__init__(spec_name)

    # --- 核心密码学原语 ---

    # 哈希函数 H (SHA3-256)
    def hash_h(self, data):
        return SHA3_256.new(data).digest()

    # 哈希函数 G (SHA3-512)
    def hash_g(self, data):
        digest = SHA3_512.new(data).digest()
        return digest[:32], digest[32:]

    # 哈希函数 J (SHAKE-256)
    def hash_j(self, data):
        return SHAKE256.new(data).read(32)

    # 伪随机函数 (PRF, 使用 SHAKE-256)
    def prf(self, eta, seed, nonce):
        return SHAKE256.new(seed + bytes([nonce])).read(64 * eta)

    # --- 序列化/反序列化与编码/解码函数 ---

    # 压缩多项式
    def compress_poly(self, d, int_vec):
        return [(((x << d) + (self.q - 1) // 2) // self.q) % (1 << d) for x in int_vec]

    # 解压缩多项式
    def decompress_poly(self, d, int_vec):
        return [(self.q * y + (1 << (d - 1))) >> d for y in int_vec]

    # 将比特流转换为字节流
    def bits_to_bytes(self, bits):
        byte_arr = bytearray(len(bits) // 8)
        for i in range(len(byte_arr)):
            val = sum(bits[i * 8 + j] << j for j in range(8))
            byte_arr[i] = val
        return bytes(byte_arr)

    # 将字节流转换为比特流
    def bytes_to_bits(self, byte_data):
        bit_arr = bytearray(8 * len(byte_data))
        for i, byte_val in enumerate(byte_data):
            for j in range(8):
                bit_arr[8 * i + j] = (byte_val >> j) & 1
        return bit_arr

    # 编码多项式
    def encode_poly(self, d, poly_vec):
        if isinstance(poly_vec[0], list):
            return b''.join(self.encode_poly(d, p) for p in poly_vec)
        mod_val = 1 << d if d < 12 else self.q
        bits = []
        for coeff in poly_vec:
            val = coeff % mod_val
            bits.extend([(val >> i) & 1 for i in range(d)])
        return self.bits_to_bytes(bits)

    # 解码多项式
    def decode_poly(self, d, byte_data):
        mod_val = 1 << d if d < 12 else self.q
        bits = self.bytes_to_bits(byte_data)
        poly = []
        idx = 0
        for _ in range(self.n):
            val = sum(bits[idx + j] << j for j in range(d))
            poly.append(val % mod_val)
            idx += d
        return poly

    # --- 多项式采样函数 ---

    # 从种子采样一个 NTT 域的多项式
    def sample_ntt(self, seed):
        xof = SHAKE128.new(seed)
        poly = []
        while len(poly) < self.n:
            chunk = xof.read(3)
            d1 = chunk[0] + 256 * (chunk[1] % 16)
            d2 = (chunk[1] // 16) + 16 * chunk[2]
            if d1 < self.q: poly.append(d1)
            if len(poly) < self.n and d2 < self.q: poly.append(d2)
        return poly

    # 根据中心二项分布 (CBD) 采样多项式
    def sample_poly_cbd(self, eta, seed):
        bits = self.bytes_to_bits(seed)
        poly = [0] * self.n
        for i in range(self.n):
            x = sum(bits[2 * i * eta: (2 * i + 1) * eta])
            y = sum(bits[(2 * i + 1) * eta: (2 * i + 2) * eta])
            poly[i] = (x - y) % self.q
        return poly

    # --- 数论变换 (NTT) 与多项式运算 ---

    # 正向 NTT
    def ntt_forward(self, poly):
        p = poly.copy()
        layer_idx = 1
        length = 128
        while length >= 2:
            start = 0
            while start < self.n:
                zeta = NTT_ZETAS[layer_idx]
                layer_idx += 1
                for j in range(start, start + length):
                    t = (zeta * p[j + length]) % self.q
                    p[j + length] = (p[j] - t) % self.q
                    p[j] = (p[j] + t) % self.q
                start += 2 * length
            length //= 2
        return p

    # 逆向 NTT
    def ntt_inverse(self, poly_ntt):
        p = poly_ntt.copy()
        layer_idx = 127
        length = 2
        f = 3303
        while length <= 128:
            start = 0
            while start < self.n:
                zeta = NTT_ZETAS[layer_idx]
                layer_idx -= 1
                for j in range(start, start + length):
                    t = p[j]
                    p[j] = (t + p[j + length]) % self.q
                    p[j + length] = (zeta * (p[j + length] - t)) % self.q
                start += 2 * length
            length *= 2
        return [(val * f) % self.q for val in p]

    # NTT 域中的多项式点对乘法
    def multiply_ntts(self, ntt1, ntt2):
        res = []
        for i in range(0, self.n, 2):
            a1, a2 = ntt1[i], ntt1[i + 1]
            b1, b2 = ntt2[i], ntt2[i + 1]
            z = NTT_ZETAS_MUL[i // 2]
            c0 = (a1 * b1 + a2 * b2 * z) % self.q
            c1 = (a1 * b2 + a2 * b1) % self.q
            res.extend([c0, c1])
        return res

    # 多项式加法
    def add_polys(self, p1, p2):
        return [(c1 + c2) % self.q for c1, c2 in zip(p1, p2)]

    # 多项式减法
    def sub_polys(self, p1, p2):
        return [(c1 - c2) % self.q for c1, c2 in zip(p1, p2)]

    # --- KEM 的三个核心公共接口 ---

    # 密钥生成函数
    def generate_keypair(self, d, z, spec_name=None):
        self._update_spec(spec_name)
        rho, sig = self.hash_g(d + bytes([self.k]))

        A = [[self.sample_ntt(rho + bytes([j, i])) for j in range(self.k)] for i in range(self.k)]

        nonce = 0
        s = [self.ntt_forward(self.sample_poly_cbd(self.eta1, self.prf(self.eta1, sig, nonce + i))) for i in
             range(self.k)]
        nonce += self.k
        e = [self.ntt_forward(self.sample_poly_cbd(self.eta1, self.prf(self.eta1, sig, nonce + i))) for i in
             range(self.k)]

        t = [e[i][:] for i in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                t[i] = self.add_polys(t[i], self.multiply_ntts(A[i][j], s[j]))

        ek_pke = self.encode_poly(12, t) + rho
        dk_pke = self.encode_poly(12, s)

        public_key = ek_pke
        secret_key = dk_pke + public_key + self.hash_h(public_key) + z
        return public_key, secret_key

    # 密钥封装函数
    def encapsulate_secret(self, pk, msg, spec_name=None):
        self._update_spec(spec_name)
        shared_secret, random_coins = self.hash_g(msg + self.hash_h(pk))

        t = [self.decode_poly(12, pk[384 * i: 384 * (i + 1)]) for i in range(self.k)]
        rho = pk[384 * self.k: 384 * self.k + 32]
        A = [[self.sample_ntt(rho + bytes([j, i])) for j in range(self.k)] for i in range(self.k)]

        nonce = 0
        r = [self.ntt_forward(self.sample_poly_cbd(self.eta1, self.prf(self.eta1, random_coins, nonce + i))) for i in
             range(self.k)]
        nonce += self.k
        e1 = [self.sample_poly_cbd(self.eta2, self.prf(self.eta2, random_coins, nonce + i)) for i in range(self.k)]
        nonce += self.k
        e2 = self.sample_poly_cbd(self.eta2, self.prf(self.eta2, random_coins, nonce))

        u = [[0] * self.n for _ in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                u[i] = self.add_polys(u[i], self.multiply_ntts(A[j][i], r[j]))
            u[i] = self.add_polys(self.ntt_inverse(u[i]), e1[i])

        mu = self.decompress_poly(1, self.decode_poly(1, msg))

        v = [0] * self.n
        for i in range(self.k):
            v = self.add_polys(v, self.multiply_ntts(t[i], r[i]))
        v = self.add_polys(self.ntt_inverse(v), e2)
        v = self.add_polys(v, mu)

        c1 = b''.join(self.encode_poly(self.du, self.compress_poly(self.du, u[i])) for i in range(self.k))
        c2 = self.encode_poly(self.dv, self.compress_poly(self.dv, v))
        ciphertext = c1 + c2

        return shared_secret, ciphertext

    # 密钥解封装函数
    def decapsulate_secret(self, sk, ciphertext, spec_name=None):
        self._update_spec(spec_name)

        dk_pke = sk[:384 * self.k]
        pk = sk[384 * self.k: 768 * self.k + 32]
        h_pk = sk[768 * self.k + 32: 768 * self.k + 64]
        z = sk[768 * self.k + 64:]

        c1_offset = 32 * self.du * self.k
        c1 = ciphertext[:c1_offset]
        c2 = ciphertext[c1_offset:]

        u_prime = [
            self.decompress_poly(self.du, self.decode_poly(self.du, c1[32 * self.du * i: 32 * self.du * (i + 1)])) for i
            in range(self.k)]
        v_prime = self.decompress_poly(self.dv, self.decode_poly(self.dv, c2))
        s = [self.decode_poly(12, dk_pke[384 * i: 384 * (i + 1)]) for i in range(self.k)]

        w = [0] * self.n
        for i in range(self.k):
            w = self.add_polys(w, self.multiply_ntts(s[i], self.ntt_forward(u_prime[i])))

        w = self.sub_polys(v_prime, self.ntt_inverse(w))
        m_prime = self.encode_poly(1, self.compress_poly(1, w))

        k_prime, r_prime = self.hash_g(m_prime + h_pk)

        # 隐式拒绝：重新加密并比较密文，若不一致则返回衍生自z的密钥
        _, c_prime = self.encapsulate_secret(pk, m_prime, spec_name)

        if c_prime == ciphertext:
            return k_prime
        else:
            return self.hash_j(z + ciphertext)


# 程序主入口：运行FIPS 203的已知答案测试（KAT）以进行自检
if __name__ == '__main__':
    # 创建 ML_KEM 实例
    ml_kem_instance = ML_KEM()
    # 调用测试框架，对当前实现进行完整性验证
    test_all(
        ml_kem_instance.generate_keypair,
        ml_kem_instance.encapsulate_secret,
        ml_kem_instance.decapsulate_secret,
        '(ml_kem.py)'
    )