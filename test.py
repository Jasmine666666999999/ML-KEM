# 该文件是ML-KEM实现的测试框架，用于验证其是否符合FIPS 203标准。
# 它通过读取NIST提供的已知答案测试（KAT）向量文件来工作。

import json
import sys


# 辅助函数：解析JSON格式的测试向量文件。
def parse_json_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        return json.loads(content)
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}", file=sys.stderr)
        sys.exit(1)


# 准备密钥生成（KeyGen）的测试向量。
# 它会读取请求文件和结果文件，并将它们合并成一个完整的测试用例列表。
def prepare_keygen_vectors(prompt_fn, result_fn):
    req_data = parse_json_file(prompt_fn)
    res_data = parse_json_file(result_fn)
    vectors = []
    # 使用字典快速查找测试结果
    res_map = {
        (g['tgId'], t['tcId']): t
        for g in res_data['testGroups']
        for t in g['tests']
    }
    for group in req_data['testGroups']:
        algo, tgId = group['parameterSet'], group['tgId']
        for case in group['tests']:
            tcId = case['tcId']
            result = res_map.get((tgId, tcId), {})
            case.update(result)
            case['parameterSet'] = algo
            vectors.append(case)
    return vectors


# 执行密钥生成测试，并报告结果。
def execute_keygen_tests(vectors, keygen_fn):
    failures = 0
    total = len(vectors)
    print("运行 密钥生成测试...")
    for tc in vectors:
        # 调用被测试的密钥生成函数
        public_key, secret_key = keygen_fn(bytes.fromhex(tc['d']), bytes.fromhex(tc['z']), tc['parameterSet'])
        # 将输出与预期的公钥（ek）和私钥（dk）进行比较
        if not (public_key == bytes.fromhex(tc['ek']) and secret_key == bytes.fromhex(tc['dk'])):
            failures += 1

    passed_count = total - failures
    if failures == 0:
        print(f"结果: {passed_count}/{total} 全部通过")
    else:
        print(f"结果: {passed_count} 通过, {failures} 失败")
    return failures


# 准备密钥封装（Encapsulation）和解封装（Decapsulation）的测试向量。
def prepare_encdec_vectors(prompt_fn, result_fn):
    req_data = parse_json_file(prompt_fn)
    res_data = parse_json_file(result_fn)
    encap_vectors, decap_vectors = [], []
    res_map = {
        (g['tgId'], t['tcId']): t for g in res_data['testGroups'] for t in g['tests']
    }
    for group in req_data['testGroups']:
        algo, func, tgId = group['parameterSet'], group['function'], group['tgId']
        for case in group['tests']:
            tcId = case['tcId']
            result = res_map.get((tgId, tcId), {})
            case.update(result)
            case['parameterSet'] = algo
            # 根据测试的功能，将用例分配到不同的列表
            if func == 'encapsulation':
                encap_vectors.append(case)
            else:
                case['dk'] = group['dk']
                decap_vectors.append(case)
    return encap_vectors, decap_vectors


# 执行密钥封装测试，并报告结果。
def execute_encaps_tests(vectors, encaps_fn):
    failures = 0
    total = len(vectors)
    print("运行 密钥封装测试...")
    for tc in vectors:
        # 调用被测试的密钥封装函数
        shared_secret, ciphertext = encaps_fn(bytes.fromhex(tc['ek']), bytes.fromhex(tc['m']), tc['parameterSet'])
        # 将输出与预期的共享密钥（k）和密文（c）进行比较
        if not (shared_secret == bytes.fromhex(tc['k']) and ciphertext == bytes.fromhex(tc['c'])):
            failures += 1

    passed_count = total - failures
    if failures == 0:
        print(f"结果: {passed_count}/{total} 全部通过")
    else:
        print(f"结果: {passed_count} 通过, {failures} 失败")
    return failures


# 执行密钥解封装测试，并报告结果。
def execute_decaps_tests(vectors, decaps_fn):
    failures = 0
    total = len(vectors)
    print("运行 密钥解封装测试...")
    for tc in vectors:
        # 调用被测试的密钥解封装函数
        shared_secret = decaps_fn(bytes.fromhex(tc['dk']), bytes.fromhex(tc['c']), tc['parameterSet'])
        # 将输出与预期的共享密钥（k）进行比较
        if not (shared_secret == bytes.fromhex(tc['k'])):
            failures += 1

    passed_count = total - failures
    if failures == 0:
        print(f"结果: {passed_count}/{total} 全部通过")
    else:
        print(f"结果: {passed_count} 通过, {failures} 失败")
    return failures


# 主测试函数，协调并运行所有的测试流程。
def test_all(k_fn, e_fn, d_fn, iut_id=''):
    separator = "=" * 39

    print("开始进行 ML-KEM 测试")
    print(separator)

    # 准备密钥生成测试数据
    keygen_vecs = prepare_keygen_vectors(
        'json/ML-KEM-keyGen-FIPS203/prompt.json',
        'json/ML-KEM-keyGen-FIPS203/expectedResults.json'
    )
    # 准备封装/解封装测试数据
    encap_vecs, decap_vecs = prepare_encdec_vectors(
        'json/ML-KEM-encapDecap-FIPS203/prompt.json',
        'json/ML-KEM-encapDecap-FIPS203/expectedResults.json'
    )

    # 依次执行各类测试并累计失败次数
    total_failures = 0
    total_failures += execute_keygen_tests(keygen_vecs, k_fn)
    print(separator)

    total_failures += execute_encaps_tests(encap_vecs, e_fn)
    print(separator)

    total_failures += execute_decaps_tests(decap_vecs, d_fn)
    print(separator)

    # 打印最终的测试总结报告
    total_tests = len(keygen_vecs) + len(encap_vecs) + len(decap_vecs)
    if total_failures == 0:
        print(f"测试总结: 所有 {total_tests} 个测试均已通过。")
    else:
        print(f"测试总结: {total_tests} 个测试中，有 {total_failures} 个失败。")
    print(separator)


# 程序主入口
if __name__ == '__main__':
    # 从ml_kem.py导入被测试的类
    from ml_kem import ML_KEM

    # 创建一个ML_KEM实例
    kem = ML_KEM()
    # 将实例的方法传入测试框架并开始测试
    test_all(
        kem.generate_keypair,
        kem.encapsulate_secret,
        kem.decapsulate_secret
    )