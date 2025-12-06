# Checklist
## Design 

- [x] 1. Research Agent
- [x] 2. Theory Agent
- [x] 3. Architect Agent
- [ ] 4. Paper Agent
- [ ] 5. Coder Agent
- [ ] 6. Paper Refiner Agent
- [x] 7. Reviewer Agent
- [x] 8. Translator Agent

## Problems

- [x] 1. agent与prompt:json_scheme，只有research生成json
- [x] 2. thoery的review给了意见吗
- [x] 3. 阶段读取json或db完全隔离
- [ ] 4. 深入挖掘每一个模块 提示词工程 theorist阶段偶现一次不如一次，是否小轮给了修改意见，可考虑设计两种重复轮数模式：1.递进深入挖掘 2. 重写
- [ ] 5. 阶段反馈机制，完善前一个阶段的产物
- [x] 6. md.j2 模板问题
- [x] 7. architect 不仅为代码铺垫，也为论文铺垫
- [x] 8. architect 每轮完全一样
- [ ] 9. 自主选择阶段是否重写跳过
- [ ] 10. 代码写不出来就不让他硬写，做一个反馈重写idea或者theory或者architect方案
- [ ] 11. 确定review记录在json中，且被程序读取，优化md中的review展示
- [ ] 12. paper 人工下载更智能
- [ ] 13. 调研论文部分进一步优化，可以根据初始config、人工或自动review意见，动态调整调研论文领域、年份、范围等
- [ ] 14. log 输出完整，输出tokens数量

1. Research 
2. Theory
3. Architect
4. Paper
5. Coder
6. Paper Refine

src/
├── core/
│   ├── state.py       # [核心] 定义 ProjectState 庞大的 Pydantic 结构
│   ├── manager.py     # [核心] 负责 Load/Save/Merge 状态
│   ├── lifecycle.py   # [新增] 定义 Phase 的基类，规范化 run -> review -> save 流程
│   └── schema.py      # 只放细粒度的数据定义 (如 TechnicalGap)
├── agents/            # 只放 Prompt 拼接和 LLM 调用逻辑
├── tools/             # 纯工具 (PDF, Search)
└── utils/             # Log, Config