# 该文件用于对 ML-KEM 实现的性能进行基准测试。
# 它会测量密钥生成、封装、解封装操作的耗时以及相关数据的大小。

import os
import timeit
from ml_kem import ML_KEM


# 主函数，负责执行整个基准测试流程。
def run_benchmark():
    # 初始化KEM方案
    kem = ML_KEM()
    results = []
    # 定义要测试的ML-KEM安全级别
    SPEC_NAMES = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
    # 定义每个操作的重复次数以获取平均值
    ITERATIONS = 100

    print(f"开始性能测试，每个操作将重复 {ITERATIONS} 次...")

    # 遍历每个安全级别进行测试
    for spec_name in SPEC_NAMES:
        print(f"\n正在测试: {spec_name}")
        # 生成必要的随机输入数据
        d, z, msg = os.urandom(32), os.urandom(32), os.urandom(32)

        # 测试密钥生成性能，并计算平均耗时（毫秒）
        t_keygen = (timeit.timeit(lambda: kem.generate_keypair(d, z, spec_name), number=ITERATIONS) / ITERATIONS) * 1000
        # 实际生成一次密钥对，用于后续测试
        pk, sk = kem.generate_keypair(d, z, spec_name)

        # 测试密钥封装性能
        t_encaps = (timeit.timeit(lambda: kem.encapsulate_secret(pk, msg, spec_name),
                                  number=ITERATIONS) / ITERATIONS) * 1000
        # 实际执行一次封装，以获取密文等数据
        ss_orig, ct = kem.encapsulate_secret(pk, msg, spec_name)

        # 测试密钥解封装性能
        t_decaps = (timeit.timeit(lambda: kem.decapsulate_secret(sk, ct, spec_name),
                                  number=ITERATIONS) / ITERATIONS) * 1000

        # 将该安全级别的测试结果存入列表
        results.append({
            "spec": spec_name,
            "t_keygen_ms": t_keygen,
            "t_encaps_ms": t_encaps,
            "t_decaps_ms": t_decaps,
            "size_pk_b": len(pk),
            "size_sk_b": len(sk),
            "size_ct_b": len(ct),
        })
        print(f"测试完成: {spec_name}")

    # --- 结果格式化与打印 ---

    print("\n\n--- 性能测试最终结果 ---\n")

    # 定义表格的表头
    headers = [
        "    参数集   ",
        " 密钥生成 (ms)",
        "  封装 (ms)  ",
        " 解封装 (ms) ",
        " 公钥 (Bytes)",
        "私钥 (Bytes)",
        " 密文 (Bytes) "
    ]

    # 准备表格的数据行
    data_rows = [
        [
            f"{res['spec']}",
            f"{res['t_keygen_ms']:.3f}",
            f"{res['t_encaps_ms']:.3f}",
            f"{res['t_decaps_ms']:.3f}",
            str(res['size_pk_b']),
            str(res['size_sk_b']),
            str(res['size_ct_b']),
        ] for res in results
    ]

    # 定义列宽和对齐方式
    column_widths = [12] * 7
    alignments = ['left'] + ['right'] * 6

    # 辅助函数，用于打印格式对齐的一行
    def print_row(row_data, col_widths, aligns):
        cells = []
        for i, cell in enumerate(row_data):
            padding = ' ' * (col_widths[i] - len(cell))
            if aligns[i] == 'left':
                cells.append(cell + padding)
            else:
                cells.append(padding + cell)
        print(" | ".join(cells))

    # 定义分隔线
    separator_line = "-+-".join("-" * width for width in column_widths)

    # 打印表格的顶部边框、表头和分隔线
    print(separator_line)
    print(" | ".join(headers))
    print(separator_line)

    # 循环打印每一行数据
    for row in data_rows:
        print_row(row, column_widths, alignments)

    # 打印表格的底部边框
    print(separator_line)


# 程序主入口
if __name__ == '__main__':
    # 执行基准测试
    run_benchmark()