import json

from typing import Any, Dict, List, Optional


class PromptRepository:
    MOCK_PROMPT = '''
你现在是一个“工具函数生成器”。

我会提供一组工具定义。你的任务不是实现真实外部能力，而是为每一个工具生成一个“本地可执行函数”，用于联调、验证和代码运行。

【核心目标】
请为每一个工具合成一个可执行函数。由于真实输入空间无限大，你不需要覆盖所有输入，只需要：
1. 为每个工具挑选若干组“你能确定正确”的输入，优先覆盖真实场景下最常见的查询/筛选路径；
2. 对这些固定输入返回固定输出；
3. 对所有未覆盖输入，也要返回“像真实工具一样可消费的失败结果”，而不是直接抛出异常中断流程；
4. 保证代码可以直接运行；
5. 保证返回值结构与该工具的接口风格一致。

【实现要求】
1. 使用 Python 实现。
2. 每个工具生成一个独立函数，函数名尽量与工具名对应。
3. 每个函数内部只做“固定输入 -> 固定输出”的映射，不要调用外部网络、数据库、系统命令或真实 API。
4. 必须保证 deterministic（同样输入永远返回同样输出）。
5. 对于未覆盖输入，默认不要抛出 `NotImplementedError`、`ValueError`，也不要返回生硬的占位式报错。
6. 对于未覆盖输入，统一返回“查询失败 / 未查询到 / 暂无结果 / 处理失败但可继续”的结果对象，风格尽量贴近真实工具返回。推荐在返回中包含下列语义字段中的一部分：
   - `success`: `False`
   - `found`: `False`
   - `status`: 例如 `"not_found"`、`"no_result"`、`"failed"`、`"unavailable"`
   - `message`: 例如 `"未查询到相关结果"`、`"查询失败，请检查输入后重试"`、`"暂无可用数据"`
   - `data` / `result` / `items`: 与该工具风格一致的默认值（允许为空，但不能长期只给空壳）
6.1 每个工具至少提供 3 组“成功命中样例”，且这些样例应覆盖不同参数组合或不同业务分支；不要只做 1 个成功样例。
6.2 每个工具至少提供 2 组“失败但有信息量的样例”（如 not_found、invalid_filter、out_of_range），并在返回中体现失败原因与可恢复提示。
6.2.1 成功样例不仅要数量足够，还要尽量能够支撑多步 agent 任务：优先覆盖可用于“先定位对象、再筛选候选、再核验约束、最后执行或回填结果”的关键查询路径；若某工具常作为上游检索或下游核验环节，请让成功样例中的关键字段足以支持后续步骤继续使用。
6.2.2 对于失败样例，不要只给同一种失败；应尽量区分不同失败类型，并在返回中明确体现失败原因。优先覆盖下列类型中的至少 2 类，若工具语义允许则尽量覆盖更多：
   - `not_found`: 对象不存在、未查到结果、候选为空
   - `invalid_filter`: 筛选条件不合法、不受支持或无法解析
   - `missing_required`: 缺少必需参数、关键信息不足
   - `conflict`: 条件冲突、状态冲突、候选不唯一或资源不可同时满足
6.3 严禁把绝大多数输入都简单兜底成 `[]`、`""`、`{}`、`None`。未命中时可为空，但必须同时返回有语义的状态字段和消息字段，让调用方能区分“无数据”“参数问题”“服务不可用”等不同失败类型。
6.4 若工具属于列表/检索类接口，成功样例中的 `items` 不应总是单条；至少部分样例返回 2~5 条结构化记录，且字段合理。
6.5 若工具属于详情/查询类接口，成功样例中的 `result`/`data` 应包含多个关键字段（如 id/name/time/status/value），不能仅返回一个空字符串或极简占位。
7. 即使参数取值未命中、参数名有偏差、请求条件不完整，或调用方传入了不理想的内容，也优先返回自然、稳定、可继续处理的失败结果，不要暴露内部规则、固定样例、预置组合、占位实现等信息。
8. 只有在“输入不是可解析对象、类型完全不符合函数签名、运行到无法构造任何合理返回值”的极少数情况下，才允许抛出 `ValueError`；即便如此，错误信息也必须像真实服务报错，例如：
   - `"request parameters are invalid"`
   - `"unable to process the current request"`
   - `"bad request"`
   不要出现任何与占位实现、预置样例或内部兜底逻辑相关的措辞。
9. 如果一个工具天然像搜索、检索、查询、列表、详情、识别、转换、解析、推荐、统计等真实接口，那么请优先按照真实接口回包风格生成：
   - 命中已知样例时返回成功结果；
   - 未命中样例时返回失败但结构稳定的结果；
   - 不要让调用方因为“查不到”就直接 crash。
9.1 对命中样例，优先返回“有业务信息密度”的内容（例如多字段记录、可用于后续筛选/比对/聚合的字段），避免只有空值或单字段占位。
10. 如果一个工具输入是 JSON/dict，则优先基于“规范化后的 JSON 字符串”或关键字段匹配来判断。
11. 如果某些工具本身很复杂，也不要省略，至少给出 3~5 组固定样例（成功与失败都要覆盖）。
12. 所有函数放在同一个 Python 文件中。
13. 生成简单测试代码，使用 `assert` 校验固定成功样例；同时至少补充少量断言，校验未覆盖输入会返回结构化失败结果，而不是暴露内部实现痕迹。
13.1 测试代码中必须包含一个或多个 `def test_*` 函数，作为可执行测试入口。
13.2 测试中需要显式断言：成功样例返回的关键字段非空且类型正确；失败样例包含 `status` 或 `message` 等可解释信号，不是纯空列表/空字符串。
14. 对返回值风格的额外要求：
   - 若工具描述看起来像 OpenAI function/tool 调用结果，优先返回 `dict`
   - 若是列表型接口，可返回 `{"success": True/False, "items": [...], "total": n, "message": "..."}`
   - 若是详情型接口，可返回 `{"success": True/False, "result": {...} 或 None, "message": "..."}`
   - 若是转换/计算型接口，在无法处理时返回 `{"success": False, "input": 原输入, "result": None, "message": "..."}`
   - 若是 OCR/抽取/识别型接口，在失败时返回 `{"success": False, "extracted": None 或 {}, "message": "..."}`
15. 所有失败返回都必须 deterministic，且文案自然，不能包含明显暴露占位实现、固定样例、内部兜底规则、预置组合限制的措辞。
16. 输出代码中的函数返回必须让调用方感觉自己在和真实工具交互，而不是和占位实现交互。

【输出格式要求】
你必须严格只输出以下两段内容，不能多写任何解释、说明、前言、后记。

第一段：正式函数代码区
必须以这一行开始：
===PYTHON_MODULE_START===

必须以这一行结束：
===PYTHON_MODULE_END===

第二段：测试代码区
必须以这一行开始：
===PYTHON_TEST_START===

必须以这一行结束：
===PYTHON_TEST_END===

【强制格式要求】
1. 两段都必须是纯 Python 代码内容，不要再嵌套 markdown 代码块。
2. 输出中除这四个标记和两段 Python 代码外，不能有任何其他文字。
3. 正式函数代码区必须可以单独保存为一个 `.py` 文件并被 import。
4. 测试代码区必须可以单独提取出来用于测试。
5. 整个回答中，这四个标记各自只能出现一次。

下面是工具定义：
'''.strip()

    DIFFICULT_PROMPT = '''
请基于你生成的工具集合，设计一道“必须依赖这些能力才能得到答案”的标准问答题。

这里的“难”指的是：需要在同一个真实任务中做信息定位、条件筛选、一步到两步的组合推理；不是把多个无关领域的事实强行拼在一起。

要求如下：
1. 只生成 1 道题。
2. `PROMPT` 必须只包含一段自然、真实的用户提问，像聊天或业务场景里的正常询问，不要像测试脚本，不要出现“工具”“函数”“接口”“调用”等字样。
3. 问题必须围绕同一个主题、对象或任务场景展开，所有查询步骤都要服务于同一个明确目标。
4. 问题不能只靠常识直接回答，必须结合可获取的信息或可执行能力才能得出结果。
5. 难度主要来自“筛选条件、字段理解、结果比对、一步到两步计算或归纳”，不要来自生造背景、堆砌限制或跨领域混搭。
6. 最终答案必须是简短、唯一、可验证的客观答案，例如一个数字、一个日期、一个名称，或一个非常短的结构化结果。
7. 答案必须可以被明确校验，不能是开放式回答，不能模糊，不能带解释。
8. 最多只允许 2 到 3 个紧密相关的求解步骤；不要把多个彼此独立的小问题硬拼成一道题。
9. 禁止把体育、金融、农业、地理、医疗、娱乐等无关领域强行组合；禁止为了制造难度而引入与主任务无关的背景设定。
10. 禁止无意义的字符串加工，例如截取首字母、拼接代号、强制大小写变换、凑固定花哨格式；除非这本身就是任务目标中自然且必要的一部分。
10.1 严禁生成“先查两条或多条文本（如新闻标题）再取每条首单词首字母并拼接成字符串”的题型；这类题目判定为无意义任务，必须重写。
11. 输出格式应尽量简单自然，优先使用单个值，或最多 2 个字段的短结构；不要设计冗长格式约束。
12. 不要输出解题过程，不要输出分析，不要输出额外说明。
13. 你必须严格按照我指定的三段格式输出，且字段名必须完全一致。
14. 题目应当使用部分或全部可用能力，但必须体现真实用户意图，不能显得像“为了覆盖能力而覆盖能力”。
15. PROMPT 请控制在 120 个汉字以内；如果超过，说明题目设计得过于复杂，请重写得更直接。
16. `PROMPT` 中禁止出现任何暴露生成背景的话，例如测试数据、样例数据、当前数据集、本地数据、演示环境、`run_demo` 等。
17. 不要在 `PROMPT` 里要求回答者“按某种格式作答”，不要出现“请将最终答案写成”“输出为”“返回为”等格式指令；这些只允许放在 `OUTPUT_FORMAT` 字段。
18. 题目必须确实需要借助外部能力检索或计算后才能回答。若工具之间存在依赖关系，可设计为串行；若存在可独立查询的子问题，可设计为并行；也可以是串并行混合。但这些求解结构不要明说在题面里。
19. 优先生成带有真实生活或业务语境的问题，例如查询、比对、筛选、排期、核验、推荐、定位、统计，不要生成“为了验证系统而提问”的句子。
20. `PROMPT`、`OUTPUT_FORMAT`、`ANSWER` 三者必须强绑定：题面最终要用户交付什么，`OUTPUT_FORMAT` 就只能约束那个最终交付物，`ANSWER` 也必须正好是该交付物，三者的对象类型、字段数、数量级必须一致。
21. 不要把“中间检索步骤”直接堆进 `PROMPT` 里，除非这些条件本身就是最终交付物定义的一部分。题面要聚焦最终决策目标，而不是把求解过程逐步写出来。
22. 如果 `PROMPT` 问的是单个名称/数字/日期，`OUTPUT_FORMAT` 应该就是单值格式；如果 `PROMPT` 问的是列表、多字段结果、判断结论，`OUTPUT_FORMAT` 与 `ANSWER` 也必须显式呈现列表、多字段或判断值，不能偷换成单个 `//box{值}`。
23. 在输出前先自检：仅看 `PROMPT`，人类是否能直接理解为什么最终答案应该长成 `OUTPUT_FORMAT` 规定的样子；如果不能，必须重写。

请严格按照下面格式输出：

PROMPT:
<你生成的自然语言问题>

OUTPUT_FORMAT:
<该问题要求回答者使用的最终输出格式，例如：//box{值}>

ANSWER:
<该问题唯一且可验证的标准答案，必须严格符合 OUTPUT_FORMAT 中定义的格式>
'''.strip()

    EXTRA_DIFFICULT_PROMPT = '''
请基于你刚才生成的mocked工具集合，设计一道“必须依赖多个工具通过串行、并行或组合调用后才能得到答案”的高难度标准问答题。

这里的“更难”指的是：题目必须明显依赖多工具协同求解，且至少包含一段必要的组合过程，例如先检索再筛选、先定位再比对、并行查询后汇总、或串并行混合后得到唯一结论；不是把多个无关事实硬拼在一起。

要求如下：
1. 只生成 1 道题。
2. `PROMPT` 必须只包含一段自然、真实的用户提问，像聊天或业务场景里的正常询问，不要像测试脚本，不要出现“工具”“函数”“接口”“调用”等字样。
3. 问题必须围绕同一个主题、对象或任务场景展开，所有查询和推理步骤都要服务于同一个明确目标。
4. 题目必须显著依赖多个工具能力，不能只靠单个工具或常识直接回答。
5. 题目至少需要 3 个彼此相关的求解动作，其中至少包含以下结构之一：
   - 串行：前一步结果作为后一步的必要输入；
   - 并行：需要同时查询多个相关信息后再汇总；
   - 串并行组合：既有依赖链，也有可并行的子问题。
6. 难度主要来自“信息定位、条件筛选、字段理解、结果比对、交叉核验、一步到两步计算或归纳”，不要来自生造背景、堆砌限制或跨领域混搭。
7. 最终答案必须是简短、唯一、可验证的客观答案，例如一个数字、一个日期、一个名称，或一个非常短的结构化结果。
8. 答案必须可以被明确校验，不能是开放式回答，不能模糊，不能带解释。
9. 题目应尽量使用多工具的真实协作价值，不能显得像“为了覆盖能力而覆盖能力”。
10. 绝对不要生成那种即使调用了相关函数也无法根据当前 mocked 工具返回结果得出答案的问题。
11. 你设计的问题、`ANSWER` 与 `OUTPUT_FORMAT`，必须能够被当前你已经生成的 mocked 函数中的固定样例唯一支撑并求解出来；如果 mock 数据不足以支撑，就必须重写题目。
12. 若某一步所需信息在当前 mocked 返回中不可得、不可唯一确定、或无法与其他结果对齐，请不要使用该思路，改为设计一个能被现有 mocked 结果完整解出的题目。
13. 禁止依赖 mock 中不存在的隐含知识、额外背景、常识补全或未返回字段做跳步推断。
14. 禁止把体育、金融、农业、地理、医疗、娱乐等无关领域强行组合；禁止为了制造难度而引入与主任务无关的背景设定。
15. 禁止无意义的字符串加工，例如截取首字母、拼接代号、强制大小写变换、凑固定花哨格式；除非这本身就是任务目标中自然且必要的一部分。
15.1 严禁生成“先查两条或多条文本（如新闻标题）再取每条首单词首字母并拼接成字符串”的题型；这类题目判定为无意义任务，必须重写。
15.2 如果题目核心只是对查询结果做最大值、最小值、计数、拼接、排序、取最后一个、取唯一剩余项、简单求和或类似浅层加工，而没有体现真实决策目标、约束核验或执行意图，则必须重写。
16. 题目中的每一步都必须能映射回真实用户会提出的 agent 子目标，例如确认对象、定位候选、核验约束、执行操作、回填结果；不能只是为了制造唯一答案而设计中间步骤。
17. 在生成最终题目前，你必须先在内部完成一份解题计划，但不要把计划输出给我。该计划至少要覆盖：
   - 需要哪些工具；
   - 每一步输入来自哪里；
   - 少任意一步是否仍然可答；
   - 最终答案为什么唯一。
   若其中任一点不成立，就必须重写题目。
18. 优先生成 action-oriented 的真实任务，不要生成偏 factoid QA 的题。更优先的问题形态包括：筛选候选并确认联系人、定位资源并核验约束、找到目标对象后执行通知或提交动作、确认状态后决定下一步操作。应尽量避免“哪个最大”“最后一个代码是什么”“纬度之和是多少”这类以结果加工为主的问法。
19. 输出格式应尽量简单自然，优先使用单个值，或最多 2 个字段的短结构；不要设计冗长格式约束。
20. 不要输出解题过程，不要输出分析，不要输出额外说明，也不要输出你内部生成的解题计划。
21. 你必须严格按照我指定的三段格式输出，且字段名必须完全一致。
22. `PROMPT` 请控制在 150 个汉字以内；如果超过，说明题目设计得过于复杂，请重写得更直接。
23. `PROMPT` 中禁止出现任何暴露生成背景的话，例如测试数据、样例数据、当前数据集、本地数据、演示环境、`run_demo`、mock、mocked 等。
24. 不要在 `PROMPT` 里要求回答者“按某种格式作答”，不要出现“请将最终答案写成”“输出为”“返回为”等格式指令；这些只允许放在 `OUTPUT_FORMAT` 字段。
25. 优先生成带有真实生活或业务语境的问题，例如查询、比对、筛选、排期、核验、推荐、定位、统计，不要生成“为了验证系统而提问”的句子。
26. 在生成最终结果前，你必须先自检：当前 mocked 工具样例是否真的足以完整解题、是否确实需要多工具、是否符合 action-oriented 任务形态，以及答案是否唯一；若任一条件不满足，就重写。
27. `PROMPT`、`OUTPUT_FORMAT`、`ANSWER` 必须是强约束关系，不允许松散拼接。题面最终要求返回什么对象、几个对象、几个字段，输出格式和标准答案就必须一一对应。
28. 不要把“先查A、再筛B、最后核验C”这种中间过程原样塞进题面，再只让答案输出其中一个碎片结果。中间步骤可以隐含存在，但题面应聚焦最终交付物。
29. 如果题面最终要的是列表、候选集、双字段结果、状态判断或带条件的结论，`OUTPUT_FORMAT` 和 `ANSWER` 必须完整承载这些内容；不能用与题面目标弱相关的单值 `//box{...}` 草率收尾。
30. 在输出前先做一致性检查：只看 `PROMPT` 与 `OUTPUT_FORMAT`，应能清楚推断标准答案为何是该结构；若答案结构像是“另一个问题的答案”，必须重写。

请严格按照下面格式输出：

PROMPT:
<你生成的自然语言问题>

OUTPUT_FORMAT:
<该问题要求回答者使用的最终输出格式，例如：//box{值}>

ANSWER:
<该问题唯一且可验证的标准答案，必须严格符合 OUTPUT_FORMAT 中定义的格式>
'''.strip()

    BOUNDARY_MISSING_PROMPT = '''
请基于你刚才生成的工具集合，设计一道“边界行为评测题”，用于评估模型在以下两类场景中的决策是否正确：
- Missing Parameters（缺失参数）：工具能力可满足任务，但当前信息不足，正确行为是先反问澄清，不调用工具。
- Missing Functions（缺失函数）：当前工具能力根本无法满足任务，正确行为是得体拒绝并解释边界，不调用工具。

要求如下：
1. 只生成 1 道题，且只属于上述两类之一，不要混合。
2. `PROMPT` 必须是自然真实的用户提问，不要出现“工具”“函数”“接口”“调用”“评测”等字样。
3. 若你选择 Missing Parameters：题目必须明显缺少至少一个关键必需信息（如日期、地点、对象、范围、ID 等），并且该信息在不臆测的前提下无法补全。
4. 若你选择 Missing Functions：题目必须明显超出当前可用能力边界，不能通过“强行套用现有能力”解决。
5. 两类场景都必须满足：在当前轮次都不应触发任何工具调用。
6. 严禁让模型脑补参数、捏造不存在的函数、或把不匹配的能力硬凑成可解。
7. 题目必须简洁，`PROMPT` 控制在 120 个汉字以内。
8. 禁止无意义字符串加工（如首字母拼接、代号拼接、大小写游戏）；尤其禁止“查标题后取首字母拼接”的题型。
9. 不要输出解题过程，不要输出分析，不要输出额外说明。
10. 你必须严格按照三段格式输出，字段名必须完全一致。
11. `OUTPUT_FORMAT` 必须使用下面这个固定格式（用于数据标识）：
//diag{tag=BOUNDARY_CASE_V1;case=<missing_parameters|missing_functions>;expected_action=<clarifying_question|graceful_decline>;answer=<TBD>}
12. `ANSWER` 必须填充同结构，并保留占位答案，不需要真实解题值：
//diag{tag=BOUNDARY_CASE_V1;case=<...>;expected_action=<...>;answer=<TBD>}
13. `case` 与 `expected_action` 的映射必须一致：
- missing_parameters -> clarifying_question
- missing_functions -> graceful_decline
14. 你生成的题目必须能让标注者仅凭题面和工具能力边界，明确判断它属于哪一类，不得含糊。
15. `PROMPT` 中禁止暴露生成背景，例如测试数据、样例数据、本地数据、演示环境、run_demo、mock、mocked。

请严格按照下面格式输出：

PROMPT:
<你生成的自然语言问题>

OUTPUT_FORMAT:
//diag{tag=BOUNDARY_CASE_V1;case=<missing_parameters|missing_functions>;expected_action=<clarifying_question|graceful_decline>;answer=<TBD>}

ANSWER:
//diag{tag=BOUNDARY_CASE_V1;case=<...>;expected_action=<...>;answer=<TBD>}
'''.strip()

    EASY_PROMPT = '''
请基于我提供的工具集合，设计一道“必须依赖这些能力才能得到答案”的标准问答题。

要求如下：
1. 只生成 1 道题。
2. `PROMPT` 必须只包含一段自然、真实的用户提问，像聊天或业务场景里的正常询问，不要像测试脚本，不要出现“工具”“函数”“接口”“调用”等字样。
3. 问题必须围绕同一个主题、对象或任务场景展开，不要跨领域拼接无关信息。
4. 问题不能只靠常识直接回答，必须结合可获取的信息或可执行能力才能得出结果。
5. 最终答案必须是简短、唯一、可验证的客观答案，例如一个数字、一个日期、一个名称，或一个非常短的结构化结果。
6. 答案必须可以被明确校验，不能是开放式回答，不能模糊，不能带解释。
7. 问题难度适中，不要设计成超长推理题，也不要依赖主观判断。
8. 最多只允许 1 到 2 个紧密相关的求解步骤，不要堆砌条件，不要做无意义字符串拼接。
8.1 严禁生成“先查两条或多条文本（如新闻标题）再取每条首单词首字母并拼接成字符串”的题型；这类题目判定为无意义任务，必须重写。
9. 输出格式应尽量简单自然，优先使用单个值，或最多 2 个字段的短结构。
10. 不要输出解题过程，不要输出分析，不要输出额外说明。
11. 你必须严格按照我指定的三段格式输出，且字段名必须完全一致。
12. 题目应当使用部分或全部可用能力，但必须体现真实用户意图，不能显得像“为了覆盖能力而覆盖能力”。
13. `PROMPT` 中禁止出现任何暴露生成背景的话，例如测试数据、样例数据、当前数据集、本地数据、演示环境、`run_demo` 等。
14. 不要在 `PROMPT` 里要求回答者“按某种格式作答”，不要出现“请将最终答案写成”“输出为”“返回为”等格式指令；这些只允许放在 `OUTPUT_FORMAT` 字段。
15. 题目必须确实需要借助外部能力检索或计算后才能回答；根据工具关系，可以是串行、并行或串并行混合求解，但不要把这种结构直接写进题面。
16. 优先生成带有真实生活或业务语境的问题，例如查询、比对、筛选、排期、核验、推荐、定位、统计，不要生成“为了验证系统而提问”的句子。
17. `PROMPT`、`OUTPUT_FORMAT`、`ANSWER` 三者必须紧密对应：题面问什么最终结果，输出格式就约束什么结果，标准答案就完整回答什么结果。
18. 不要把与最终输出弱相关的中间步骤写成题面主体，也不要让 `OUTPUT_FORMAT` 和 `ANSWER` 只覆盖题面中的一小部分信息。
19. 如果题面要求的是多个对象、多个字段或判断结论，`OUTPUT_FORMAT` 和 `ANSWER` 也必须明确体现相同的数量和结构；不能把多项任务偷换成单个 `//box{值}`。

请严格按照下面格式输出：

PROMPT:
<你生成的自然语言问题>

OUTPUT_FORMAT:
<该问题要求回答者使用的最终输出格式，例如：//box{值}>

ANSWER:
<该问题唯一且可验证的标准答案，必须严格符合 OUTPUT_FORMAT 中定义的格式>
'''.strip()

    MOCK_ORIGINAL_QUESTION_GUIDANCE = '''
【原始问题增强要求】
下面我会额外提供一条原始用户问题，目的是帮助你理解这些工具最可能被怎样组合使用。
请据此优化你生成的函数和测试样例：
1. 优先覆盖原始问题中最核心、最可能出现的参数组合与查询路径。
2. 如果原始问题涉及多步查询，请尽量让固定样例能支撑这类串行、并行或串并行混合求解。
3. 仍然只生成 deterministic 的固定映射，不要实现真实外部能力。
4. 原始问题只是帮助你挑选更贴近真实场景的输入输出，不代表你必须逐字复现其中每个字段。
5. 不要在输出代码或错误信息中泄露“原始问题增强”“参考问题”等字样。
'''.strip()

    QA_ORIGINAL_QUESTION_GUIDANCE = '''
【原始问题增强要求】
下面我会额外提供一条原始用户问题，目的是帮助你生成更高质量的新题目。
请严格遵守以下要求：
1. 将它视为“语义风格与真实任务意图”的参考，而不是要你原样改写。
2. 你生成的是一道新的题目，必须仍然以我给定的 `PROMPT / OUTPUT_FORMAT / ANSWER` 三段格式输出。
3. 新题目应尽量继承原始问题的真实感、任务目标和信息组织方式，但要结合当前你已经生成的 mock 工具能力，确保答案可被当前 mock 数据唯一支撑。
4. 允许你对原始问题做收缩、重组、具体化或轻度改写，让它更适合产出唯一、可校验、便于 RL 使用的最终答案。
5. 如果原始问题本身没有约束最终答案格式，请你自行把结果收束成简短唯一的目标，并把格式要求只写在 `OUTPUT_FORMAT` 中，不要写进 `PROMPT`。
6. 不要在 `PROMPT` 中提及原始问题、mock、测试、样例、数据集等生成背景。
'''.strip()

    QA_MOCK_CONTEXT_GUIDANCE = '''
【当前 mocked 环境约束】
下面我会额外提供当前已经生成好的 mocked 函数摘要。
请严格基于这些已存在的 mocked 能力来设计题目，并遵守：
1. 你只能依赖摘要中已经明确展示的函数、参数和可返回样例语义；不要臆造新字段、新能力或额外知识。
2. 题目必须能被当前 mocked 环境唯一支撑并求解；若 mock 信息不足，就重写题目。
3. 若某个思路只依赖单个函数即可完成，而当前模式要求更高难度或多工具协作，请重写题目。
4. 不要在 `PROMPT` 中泄露 mock、mocked、样例、测试、代码、函数摘要等背景信息。
5. 题面里的每个显式要求都必须在最终答案中有对应落点；不要写很多中间查询动作，却只在 `ANSWER` 中给出一个弱相关碎片。
6. 优先把多步能力收束成一个清晰的最终交付物，再让 `OUTPUT_FORMAT` 和 `ANSWER` 与这个交付物严格对齐。
'''.strip()

    FILTER_REVIEW_PROMPT = '''
你是一个用于 agent 合成数据的终审质检员。你的任务不是改写样本，而是判断这个样本是否值得保留给后续训练或评测使用。

请围绕下面五个维度做严格评审：
1. `mock_quality`
   - mocked module 是否像“可消费的本地工具”，而不是明显占位实现。
   - 是否存在明显暴露内部实现的话术，例如 “unsupported mocked input”“No mock defined”“only specific predefined combinations are supported”。
   - 是否大量使用 `[]`、`""`、`{}`、`None` 之类空壳返回来敷衍未命中输入。
   - 测试是否具备基本有效性，例如包含 `test_*` 入口、存在断言、不是空测或纯形式测。
2. `qa_consistency`
   - `PROMPT` / `OUTPUT_FORMAT` / `ANSWER` 是否三者强绑定、对象一致、结构一致。
   - 如果题目问的是多项结果、多字段结果或列表，`OUTPUT_FORMAT` 不能偷换成单值 `//box{...}`。
   - `ANSWER` 必须严格符合 `OUTPUT_FORMAT`，且不能与 `PROMPT` 基本相同。
3. `answerability`
   - 题目是否真的能被当前 mocked 能力唯一支撑并求解，不能依赖 mock 中不存在的字段、隐含知识或额外常识补全。
   - extra_difficult / difficult 样本要体现真实的多步查询、筛选、核验或组合求解价值，而不是表面多步、实际单步可答。
   - boundary_missing 样本必须真的是“缺参数先澄清”或“缺能力应拒绝”，且 `//diag{...}` 的 case / expected_action 映射正确。
4. `task_realism`
   - 题目是否像真实用户目标，而不是为了覆盖工具而覆盖工具。
   - 严禁无意义任务，例如“查多条标题后取首字母拼接字符串”“只做 acronym/首字母拼接”“只有浅层字符串游戏或无业务意义的机械加工”。
   - 尽量偏真实的查询、筛选、定位、核验、排期、推荐、统计、执行/回填类任务。
5. `mode_fit`
   - `PROMPT` 长度、题目复杂度、结构与 `qa_mode` 是否匹配。
   - `PROMPT` 中不得泄露生成背景，例如测试数据、样例数据、当前数据集、本地数据、演示环境、run_demo、mock、mocked、工具调用、函数调用、评测等。

请按下面原则给结论：
- 只要存在结构性硬伤，就判 `reject`。例如：题目不可解、格式不一致、答案不匹配、边界类型错、明显无意义任务、mock/test 明显占位或极差。
- 只有当样本整体可运行、可解、结构自洽、任务真实时，才判 `accept`。
- 不要因为轻微措辞、命名风格、字段顺序等小问题就拒绝；重点抓会伤害数据质量的核心问题。

你的输出必须是严格 JSON，不能带 markdown，字段必须完整：
{
  "decision": "accept" 或 "reject",
  "confidence": 0 到 100 的整数,
  "summary": "一句中文总结",
  "fatal_issues": ["..."],
  "minor_issues": ["..."],
  "dimension_scores": {
    "mock_quality": 1 到 5 的整数,
    "qa_consistency": 1 到 5 的整数,
    "answerability": 1 到 5 的整数,
    "task_realism": 1 到 5 的整数,
    "mode_fit": 1 到 5 的整数
  }
}

如果你认为样本可接受，`fatal_issues` 必须为空数组。
'''.strip()

    AGENT_SYN_SELECT_PROMPT = '''
你是一个“agentic RL 合成种子筛选器”。

我会给你一条原始工具使用样本，包括：
- 工具定义
- 原始对话 / 工具调用轨迹
- 若干统计信息

你的任务不是评价当前 assistant 回答得好不好，而是判断：这条样本的工具集合与可观测返回字段，是否适合作为后续合成 agentic RL 数据的“种子样本”。

这里的目标数据要求非常明确：
1. 应当能合成出“有一定难度”的真实任务。
2. 解题时应当需要结合多个工具获取信息，最好包含串行依赖、并行汇总、交叉核验或筛选后确认。
3. 最终答案必须能收束成简短、唯一、可验证的 final answer ground truth。
4. 问题要像真实用户目标，而不是为了覆盖工具而拼凑。

请重点围绕下面五个维度严格评估：
1. `tool_synergy`
   - 这些工具是否围绕同一个主题/对象/任务自然协作。
   - 是否存在清晰的上游检索 -> 下游详情 / 核验 / 执行动作链路。
   - 如果工具只是堆在一起但彼此无关，或明显跨域硬拼，应低分。
2. `multi_step_depth`
   - 是否天然支持至少 2 个以上相关工具参与的多步求解。
   - 是否真的需要筛选、比对、核验、回填、确认，而不是单工具一次查询就能答完。
   - 如果本质上是单跳查一个值，或只是把多个独立小问题拼在一起，应低分。
3. `ground_truth_feasibility`
   - 从工具定义和已观测返回字段看，是否能构造“唯一、客观、短小、可校验”的最终答案。
   - 更偏好：单个名称、日期、数值、短结构化对象、明确判断结论。
   - 如果返回主要是开放式长文本、主观总结、模糊推荐、不可唯一收束的报告，应低分。
4. `synthesis_potential`
   - 是否容易基于这些工具合成新的自然问题，并给出对应 ground truth。
   - 是否能在不暴露工具背景的情况下，把中间步骤收束成一个清晰最终交付物。
   - 如果很难写出自然题面，或者题目大概率会变成开放式报告题，应低分。
5. `realism_focus`
   - 整体是否像真实 agent 任务，而不是“工具超市”式样本。
   - 若工具池过大、主题发散、跨域拼接严重、业务目标不集中，应低分。
   - 若任务目标真实清晰，例如查询、定位、筛选、核验、确认、排期、推荐后落到单一决策，更适合接受。

请特别遵守以下拒绝原则：
- 只靠单个工具或单次查询就能得到答案的样本，通常应拒绝。
- 多工具但彼此无关、靠强行跨域拼接制造复杂度的样本，应拒绝。
- 很难收束出唯一 final answer ground truth 的样本，应拒绝。
- 明显更适合做开放式 long-form assistant 回复，而不是 final-answer RL 的样本，应拒绝。
- 工具数量极多但缺乏主题聚焦、像通用杂烩 API 集合的样本，通常应拒绝。

同时也请注意：
- 原始轨迹中如果出现了错误调用、参数缺失、assistant 走偏，不应自动判死刑。
- 只要工具本身仍然能支撑“多工具、多步、可验证 final answer”的新题目，就可以接受。
- 你的判断要“主要看工具是否适合后续合成”，原始对话只作为真实任务意图与字段可观测性的参考。

请按下面原则给结论：
- 只有当样本明显适合后续合成高质量 agentic RL final-answer 数据时，才判 `accept`。
- 只要存在结构性硬伤，例如主题发散、单工具即可解、答案无法唯一验证、题面很难自然生成，就判 `reject`。

你的输出必须是严格 JSON，不能带 markdown，字段必须完整：
{
  "decision": "accept" 或 "reject",
  "confidence": 0 到 100 的整数,
  "summary": "一句中文总结",
  "fatal_issues": ["..."],
  "minor_issues": ["..."],
  "dimension_scores": {
    "tool_synergy": 1 到 5 的整数,
    "multi_step_depth": 1 到 5 的整数,
    "ground_truth_feasibility": 1 到 5 的整数,
    "synthesis_potential": 1 到 5 的整数,
    "realism_focus": 1 到 5 的整数
  },
  "suggested_task_pattern": "一句短语，描述更适合的题型；若 reject 也要给出最接近的可行方向或填 not_recommended",
  "suggested_answer_type": "single_value | short_object | short_list | binary_decision | not_recommended"
}

如果你认为样本可接受，`fatal_issues` 必须为空数组。
'''.strip()

    @classmethod
    def get_qa_prompt(cls, mode: str) -> str:
        if mode == "extra_difficult":
            return cls.EXTRA_DIFFICULT_PROMPT
        if mode == "boundary_missing":
            return cls.BOUNDARY_MISSING_PROMPT
        if mode == "difficult":
            return cls.DIFFICULT_PROMPT
        return cls.EASY_PROMPT

    @classmethod
    def build_original_question_context(cls, original_question: Optional[str]) -> str:
        if not original_question:
            return ""
        return (
            "\n\n【原始用户问题（仅作语义参考，不可照抄）】\n"
            f"{original_question.strip()}\n"
        )


def build_mock_prompt(tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = [PromptRepository.MOCK_PROMPT, "", "【可用工具】"]

    if not tools:
        lines.append("无")
        return "\n".join(lines)

    for i, tool in enumerate(tools, 1):
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        required = set(params.get("required", []))
        props = params.get("properties", {})

        lines.append(f"{i}. {name}")
        if desc:
            lines.append(f"   描述: {desc}")

        if props:
            lines.append("   参数:")
            for p_name, p_info in props.items():
                p_type = p_info.get("type", "any")
                p_desc = p_info.get("description", "")
                req_mark = " [必填]" if p_name in required else ""
                line = f"   - {p_name}: {p_type}{req_mark}"
                if p_desc:
                    line += f" - {p_desc}"
                lines.append(line)
        else:
            lines.append("   参数: 无")

    return "\n".join(lines)


def build_mock_prompt_with_context(
    tools: List[Dict[str, Any]],
    use_original_question_for_generation: bool,
    original_question: str = "",
) -> str:
    base_prompt = build_mock_prompt(tools)
    if not (use_original_question_for_generation and original_question):
        return base_prompt

    return (
        base_prompt
        + "\n"
        + PromptRepository.MOCK_ORIGINAL_QUESTION_GUIDANCE
        + PromptRepository.build_original_question_context(original_question)
    )


def build_qa_prompt_with_context(
    qa_mode: str,
    use_original_question_for_generation: bool,
    original_question: str = "",
    mock_context: str = "",
) -> str:
    base_prompt = PromptRepository.get_qa_prompt(qa_mode)
    parts = [base_prompt]

    if use_original_question_for_generation and original_question:
        parts.extend([
            PromptRepository.QA_ORIGINAL_QUESTION_GUIDANCE,
            PromptRepository.build_original_question_context(original_question),
        ])

    if mock_context:
        parts.extend([
            PromptRepository.QA_MOCK_CONTEXT_GUIDANCE,
            "\n\n【当前 mocked 函数摘要】\n" + mock_context.strip() + "\n",
        ])

    return "\n".join(parts)


def _truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(obj)


def _format_tools_for_filter(tools: Any, max_chars: int = 6000) -> str:
    if not tools:
        return "[]"
    return _truncate_text(_safe_json_dumps(tools), max_chars)


def _format_tools_for_selection(
    tools: Any,
    max_tools: int = 20,
    max_chars: int = 9000,
) -> str:
    if not isinstance(tools, list) or not tools:
        return "[]"

    names: List[str] = []
    lines: List[str] = [f"total_tools: {len(tools)}"]

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {}) if isinstance(tool.get("function", {}), dict) else {}
        name = str(fn.get("name", "")).strip()
        if name:
            names.append(name)

    if names:
        lines.append("all_tool_names: " + ", ".join(names))

    for idx, tool in enumerate(tools[:max_tools], 1):
        if not isinstance(tool, dict):
            continue

        fn = tool.get("function", {}) if isinstance(tool.get("function", {}), dict) else {}
        name = str(fn.get("name", "")).strip() or f"<tool_{idx}>"
        desc = str(fn.get("description", "")).strip()
        params = fn.get("parameters", {}) if isinstance(fn.get("parameters", {}), dict) else {}
        required = set(params.get("required", [])) if isinstance(params.get("required", []), list) else set()
        props = params.get("properties", {}) if isinstance(params.get("properties", {}), dict) else {}

        lines.append(f"{idx}. {name}")
        if desc:
            lines.append("   desc: " + _truncate_text(desc, 240))

        param_parts: List[str] = []
        for p_name, p_info in list(props.items())[:12]:
            p_type = "any"
            if isinstance(p_info, dict):
                p_type = str(p_info.get("type", "any"))
            req_mark = "*" if p_name in required else ""
            param_parts.append(f"{p_name}:{p_type}{req_mark}")

        if param_parts:
            lines.append("   params: " + ", ".join(param_parts))
        elif required:
            lines.append("   required: " + ", ".join(sorted(required)))
        else:
            lines.append("   params: <none>")

    if len(tools) > max_tools:
        lines.append(f"... omitted {len(tools) - max_tools} tools")

    return _truncate_text("\n".join(lines), max_chars)


def _format_messages_for_selection(
    messages: Any,
    max_messages: int = 18,
    max_chars: int = 9000,
) -> str:
    if not isinstance(messages, list) or not messages:
        return "<empty>"

    lines: List[str] = []
    for idx, message in enumerate(messages[:max_messages], 1):
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip() or "<unknown>"
        content = message.get("content", "")

        if isinstance(content, str):
            normalized = content
            if role in {"tool_call", "tool_response"}:
                try:
                    normalized = _safe_json_dumps(json.loads(content))
                except Exception:
                    normalized = content
        else:
            normalized = _safe_json_dumps(content)

        lines.append(f"[{idx}] role={role}")
        lines.append(_truncate_text(normalized, 900))

    if len(messages) > max_messages:
        lines.append(f"... omitted {len(messages) - max_messages} messages")

    return _truncate_text("\n\n".join(lines), max_chars)


def build_filter_prompt_with_context(record: Dict[str, Any]) -> str:
    qa = record.get("qa") or {}
    test_report = record.get("test_report") or {}
    sample_idx = record.get("sample_idx")
    qa_mode = record.get("qa_mode", "")
    original_question = record.get("original_question", "")
    mock_context = record.get("mock_context", "")
    module_code = record.get("module_code", "")
    test_code = record.get("test_code", "")
    tools = record.get("tools") or []

    parts = [
        PromptRepository.FILTER_REVIEW_PROMPT,
        "",
        "【样本元信息】",
        f"- sample_idx: {sample_idx}",
        f"- qa_mode: {qa_mode}",
        f"- generation_status: {record.get('status', '')}",
        "",
        "【原始用户问题（若有）】",
        _truncate_text(original_question or "<empty>", 1200),
        "",
        "【题目三元组】",
        "PROMPT:",
        _truncate_text(qa.get("PROMPT", ""), 2000),
        "",
        "OUTPUT_FORMAT:",
        _truncate_text(qa.get("OUTPUT_FORMAT", ""), 1200),
        "",
        "ANSWER:",
        _truncate_text(qa.get("ANSWER", ""), 1200),
        "",
        "【mock 测试报告】",
        _truncate_text(_safe_json_dumps(test_report), 3000),
        "",
        "【工具定义】",
        _format_tools_for_filter(tools),
        "",
        "【mocked 函数摘要】",
        _truncate_text(mock_context or "<empty>", 2500),
        "",
        "【module_code】",
        _truncate_text(module_code or "<empty>", 7000),
        "",
        "【test_code】",
        _truncate_text(test_code or "<empty>", 4000),
    ]
    return "\n".join(parts)


def build_agent_syn_select_prompt_with_context(record: Dict[str, Any]) -> str:
    tools = record.get("tools") or []
    messages = record.get("messages") or []
    tool_names = record.get("tool_names") or []
    observed_tool_names = record.get("observed_tool_names") or []

    parts = [
        PromptRepository.AGENT_SYN_SELECT_PROMPT,
        "",
        "【样本元信息】",
        f"- sample_idx: {record.get('sample_idx')}",
        f"- tool_definition_count: {record.get('tool_definition_count', len(tools) if isinstance(tools, list) else 0)}",
        f"- tool_call_count: {record.get('tool_call_count', 0)}",
        f"- observed_tool_count: {record.get('observed_tool_count', len(observed_tool_names) if isinstance(observed_tool_names, list) else 0)}",
        f"- tool_names: {', '.join(tool_names) if tool_names else '<empty>'}",
        f"- observed_tool_names: {', '.join(observed_tool_names) if observed_tool_names else '<empty>'}",
        "",
        "【原始首轮用户问题】",
        _truncate_text(record.get("original_question") or "<empty>", 1600),
        "",
        "【工具定义摘要】",
        _format_tools_for_selection(tools),
        "",
        "【原始轨迹摘要】",
        _format_messages_for_selection(messages),
    ]
    return "\n".join(parts)
