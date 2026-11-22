# 2026-11-22 GitHub 交付完整性检查

核查对象：`WHN123-A11Y/franka-xhand-pi05-checkpoint`；检查起点 main `9f29c118`。

## 找到并修复的问题

1. Student/K4 配套 Release 已上传但处于草稿，外部无法下载。现已发布 `student-k4-step1500`。
2. 配套下载脚本此前未提交，README 仍要求单独索取权重。已新增无需登录的下载/断点恢复脚本与固定大小、SHA-256清单，更新 VS Code 入口。
3. step1500 新GitHub档案有7分卷，原解包脚本只支持百度网盘的3分卷。现同时支持两种布局；推荐使用完整配套下载器。
4. 原生训练入口缺 `native_openpi/scripts/train.py`。已从实际训练源码复制，来源SHA记录在 `native_openpi/ADDITIONAL_SOURCE_PROVENANCE.json`。
5. 冻结Teacher清单声明的导出脚本和两个小型夹具缺失。已按原冻结SHA补齐，19项完整校验通过。

## 实际通过的检查

- 干净克隆远端源码核查及CPU导入，Student/K4核心与真实训练源比对。
- step1500 Release全部9个下载资产的远端digest/大小与本地逐字节校验一致。
- 合并7分卷通过zstd校验，19/19个策略文件哈希一致，包含归一化参数。
- 实际从GitHub下载Student/K4 ZIP，CRC及10/10文件SHA一致。
- 干净克隆目录执行恢复脚本：已验证的策略缓存加实际远端ZIP，恢复全部配套资产；另验证流式解包逻辑。大策略未重复从远端下载，远端一致性依据GitHub的SHA-256 digest。
- `python3 scripts/check_assets.py --weights-only --full-policy-hash`通过。
- 发布后使用无认证HTTP下载校验清单，SHA一致；下载无需GitHub账户。
- 独立step30000公开Release的21/21资产（17分卷+4辅助文件）与原交接清单大小、远端SHA-256和上传状态全部一致。
- Python/Shell语法检查通过；此次检查未占用消融GPU。

## 仍需单独提供或本机验证

真实记录帧、reference预测和原始训练数据未公开，须从团队交接包获取。Teacher模型另行提供。历史训练代码保留服务器路径和严格metadata路径校验，新机器重新训练需要迁移这些配置及原始数据。

本次没有在另一台机器全新安装所有依赖，也没有重跑动作推理、训练或真机闭环。文件完整性不等于这些验证已完成。

配套版本：baseline1500 / Student2000 / K4 12273；baseline30000是独立快照，不保证与此Student/K4配套。
