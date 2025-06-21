此代码为密码学进展期末设计，是很严肃的ML-KEM相关python代码实现，仅用作学习。

其中：

ml_kem.py实现了完整、标准化的 ML-KEM (Kyber) 库，覆盖 FIPS 203 规范的所有核心原语（哈希、PRF、CBD 采样、NTT、多项式运算等）及三种安全级别参数；

test.py提供了测试框架。解析 NIST KAT (JSON) 向量，依次调用被测实现的 KeyGen、Encap、Decap 接口并比对期望输出，统计通过/失败用例数量，为快速回归测试提供自动化验证环境。

benchmark.py是一个性能基准测试脚本。对 ML-KEM-512/768/1024 三个参数集分别测量密钥生成、封装、解封装的平均耗时及公钥、私钥、密文大小，帮助了解实现的效率和资源占用。

attack.py演示了 “Roulette” 故障攻击对 ML-KEM 密钥封装机制的完整实现与模拟流程。

defend.py为带“提交-频闪”防护逻辑的加固型 ML-KEM 实现。在标准流程外加入随机提交检查与频闪检测，用于抵御典型故障注入攻击，同时保留与原版兼容的接口（keygen、encrypt、decrypt），可与攻击脚本对比测试。

attack_after_defend.py是在集成了“提交-频闪”防御之后，展示改进版攻击如何绕过该防御。代码结构与 attack.py 相似，但省去了故障插针位置参数，并针对防御逻辑调整了攻击流程，用于比较防御前后的安全差异。

代码运行结果截图在pictures文件夹中，以下是简要分析。

1. 算法正确性验证

![test](https://github.com/user-attachments/assets/1e7597f8-cc0d-456b-b57a-9e9c4fdde152)

结果显示所有 180 个官方已知答案测试均已通过。这表明该 ML-KEM 实现是正确且符合规范的。

2. 算法性能基准测试

![benchmark](https://github.com/user-attachments/assets/81d2f427-5567-4eca-997e-eaa1d55a0fed)

测量 ML-KEM 在不同安全级别下的性能指标和数据大小。

3. 故障攻击

![attack1_1](https://github.com/user-attachments/assets/38f8290d-ff11-4fa5-9bea-93727082cd9b)
![attack1_2](https://github.com/user-attachments/assets/0f9d1a1c-641c-453d-a791-6e5ec89ed909)

结果表明攻击成功。在总耗时 471.24 秒后，利用 10000 个不等式完全恢复了密钥。这证明了标准 ML-KEM 实现在无特殊防护时，对此类物理攻击是脆弱的。

4. 防御机制验证

![defend](https://github.com/user-attachments/assets/ddead669-54ce-4aa0-b964-4190eee75462)

该结果展示了增加“提交-频闪”防御机制后，能够有效抵御上述的“Roulette”攻击。

5. 改进型攻击验证

![attack2](https://github.com/user-attachments/assets/4dc67c97-3b7a-485f-bc5a-2de4ab76fb24)

结果表明攻击成功，防御被绕过。服务器未能检测到篡改，使得攻击者可以继续利用此信息构建不等式来恢复私钥。
