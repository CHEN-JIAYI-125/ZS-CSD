#!/usr/bin/env python3
"""Refresh templated claim knowledge entries and remove neutral_hint from all entries."""

import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "data" / "target_knowledge_template.json"
BAD_CLAIMS_PATH = Path(__file__).resolve().parent.parent / "data" / "_bad_claims.json"

TEMPLATE_MARKERS = (
    "赞同该命题",
    "反驳该命题",
    "认为其成立、合理",
    "不合理、有害",
    "合理性、价值",
    "观点性命题，判断重点是",
    "支持反对",
    "应被尊重、理解或正常看待",
    "认为其不应被接受",
)

UPDATES = {
    "结婚一定要征得父母同意": {
        "description": "婚恋是否必须获得父母同意、家庭干预边界的公共讨论。",
        "favor_reason": "认为父母经验可把关、尊重家庭伦理或避免冲动结合。",
        "against_reason": "主张婚姻自主、反对父母包办或认为成年人有权独立决定。",
    },
    "人大代表建议把体育由副科变主科": {
        "description": "是否将体育升格为主科、与语数外同等考核的公共讨论。",
        "favor_reason": "认为可增强青少年体质、落实每天锻炼一小时、纠正重文轻体。",
        "against_reason": "认为加重课业负担、师资场地不足、主科已饱和且易流于形式。",
    },
    "应该消除对罪犯子女考公限制": {
        "description": "罪犯直系亲属报考公务员是否应取消连坐式限制的争议。",
        "favor_reason": "认为子女不应代父受过、连坐违背现代法治与机会平等。",
        "against_reason": "认为需政审把关、防范利益输送或维护公职队伍纯洁性。",
    },
    "超雄基因宝宝不应该生下来": {
        "description": "发现胎儿XYY等异常时是否应终止妊娠的伦理争议。",
        "favor_reason": "认为应尊重知情选择、减轻家庭抚养风险或基于医学建议终止。",
        "against_reason": "强调基因不等于行为、反对标签化堕胎、残障与生命权应受保护。",
    },
    "教师工作轻松": {
        "description": "关于教师职业强度与假期福利是否匹配的社会认知争议。",
        "favor_reason": "认为寒暑假长、工作稳定、相比企业压力较小。",
        "against_reason": "强调备课批改、班主任、迎检与低薪，认为并不轻松。",
    },
    "家庭教育不得有任何形式家庭暴力": {
        "description": "《家庭教育促进法》禁止家庭暴力的规范是否必要、如何界定的讨论。",
        "favor_reason": "认为应立法保护儿童、明确打骂亦属家暴、推动科学育儿。",
        "against_reason": "担心定义过宽、干预正常管教或认为应留给家庭更多空间。",
    },
    "像国外一样取消防疫措施": {
        "description": "新冠防控是否应参照部分国家全面放开、取消严格措施的争论。",
        "favor_reason": "认为应恢复经济正常生活、过度封控代价大、与病毒共存。",
        "against_reason": "强调医疗挤兑风险、老人等脆弱群体需保护、不宜照搬国外。",
    },
    "sora没什么用处": {
        "description": "OpenAI视频生成模型Sora是否被高估、有无实际产业价值的讨论。",
        "favor_reason": "认为只是演示噱头、落地难、无法替代专业影视制作。",
        "against_reason": "认可其技术突破、内容生产效率与AI视频商业化潜力。",
    },
    "反对日本动漫文化": {
        "description": "是否应限制或抵制日本动漫及二次元文化影响的公共争论。",
        "favor_reason": "认为存在不良价值观、历史议题敏感或冲击本土文化。",
        "against_reason": "认为可正常欣赏、反对文化污名化或不应因国籍一概否定。",
    },
    "反对读书无用论": {
        "description": "是否应反驳「读书无用、学历贬值」等否定教育价值的观点。",
        "favor_reason": "认为教育提升认知与机会、读书仍是阶层流动重要通道。",
        "against_reason": "认为部分行业学历回报低、成功不只靠读书或应尊重多元路径。",
    },
    "拼多多不好": {
        "description": "低价电商平台拼多多的消费体验与商业模式评价争议。",
        "favor_reason": "批评假货多、套路营销、砍一刀骚扰或损害消费尊严。",
        "against_reason": "认可其低价普惠、满足下沉市场或认为瑕不掩瑜。",
    },
    "新能源电车不好": {
        "description": "新能源电动汽车相较燃油车是否值得购买的公共评价。",
        "favor_reason": "质疑续航焦虑、充电不便、电池衰减或冬季性能差。",
        "against_reason": "认可使用成本低、环保趋势、智能化体验或政策扶持。",
    },
    "反对计划生育政策": {
        "description": "对历史上计划生育政策效果与后果的回顾性评价争议。",
        "favor_reason": "认为造成少子化、家庭创伤、性别比例失衡等长期问题。",
        "against_reason": "认为当时控制人口必要、减轻资源压力或避免更大危机。",
    },
    "反对转基因食品": {
        "description": "是否应抵制转基因食品、对其安全性持怀疑态度的争论。",
        "favor_reason": "担心长期健康风险、监管不足或对自然食品的不信任。",
        "against_reason": "认为科学评估下安全、可缓解粮食压力或反对谣言妖魔化。",
    },
    "“拍照搜题”作业app不好": {
        "description": "学生用拍照搜题App完成作业是否损害学习效果的争议。",
        "favor_reason": "认为助长抄答案、削弱独立思考、违背作业训练目的。",
        "against_reason": "认为可辅助解题、减轻家长辅导压力或提供学习参考。",
    },
    "反对中医": {
        "description": "是否应否定中医科学性、限制其中医医疗地位的公共讨论。",
        "favor_reason": "认为缺乏循证证据、易延误病情或被伪中医利用。",
        "against_reason": "认可传承经验、调理价值或认为应中西医结合而非全盘否定。",
    },
    "反对教师因同性恋身份被开除": {
        "description": "教师因性取向遭解雇是否构成歧视、应否被制止的争议。",
        "favor_reason": "认为性取向与教学能力无关、反对就业歧视与人权侵害。",
        "against_reason": "认为学校有权维护形象、担心对学生价值观影响或家长接受度。",
    },
    "反对同性恋": {
        "description": "同性恋是否应被接受、是否构成正常性取向的社会争论。",
        "favor_reason": "认为违背传统家庭伦理、不应公开宣扬或担心对青少年影响。",
        "against_reason": "主张性取向平等、反对污名化或认为属个人自由与基本权利。",
    },
    "不应限制学生带手机入校": {
        "description": "中小学是否应禁止或严格限制学生携带手机进校园的争议。",
        "favor_reason": "认为便于联系家长、应急需要或反对一刀切没收。",
        "against_reason": "认为手机分散注意力、助长沉迷或课堂秩序需严格管理。",
    },
    "反对高考移民": {
        "description": "异地高考、学籍挂靠等「高考移民」是否应被禁止的争论。",
        "favor_reason": "认为破坏教育公平、挤占本地名额或钻政策空子。",
        "against_reason": "认为人口流动应配套升学权利、或现行户籍制度本身不公。",
    },
    "排斥电车": {
        "description": "公众对电动汽车的接受度与购买意愿的负面态度讨论。",
        "favor_reason": "认为充电麻烦、续航不够、电池安全或保值率差。",
        "against_reason": "认为体验被误解、技术已改善或油电各有适用场景。",
    },
    "反对医疗集采": {
        "description": "国家药品耗材集中带量采购政策是否损害质量与供应的争议。",
        "favor_reason": "担心低价竞争影响药效、企业利润与后续研发供应。",
        "against_reason": "认为显著降药价、减轻患者负担、挤压中间环节暴利。",
    },
    "反对收彩礼": {
        "description": "婚嫁彩礼习俗是否应被废除或严格限制的公共讨论。",
        "favor_reason": "批评高额彩礼加重负担、物化婚姻或变相买卖。",
        "against_reason": "认为彩礼体现诚意、尊重女方家庭或符合地方婚俗传统。",
    },
    "反对周琦": {
        "description": "篮球运动员周琦的公众形象、国家队表现与职业选择的评价争议。",
        "favor_reason": "批评其职业态度、关键场次发挥、转会选择或自律问题。",
        "against_reason": "认可其内线天赋、防守价值或为国家队做出的贡献。",
    },
    "反对不想结婚但想要孩子的单身生育": {
        "description": "未婚女性通过辅助生殖等方式独立生育是否应被允许的讨论。",
        "favor_reason": "认为冲击传统家庭伦理、不利于孩子成长或应限制辅助生殖。",
        "against_reason": "支持女性生育自主权、认为不必以婚姻为前提也可养育孩子。",
    },
    "反对吃预制菜": {
        "description": "预制菜进入校园、餐厅是否应被抵制或严格标识的争议。",
        "favor_reason": "担心添加剂、营养流失、透明度不足或剥夺现做新鲜度。",
        "against_reason": "认为标准化安全可控、降低成本或反对妖魔化工业食品。",
    },
    "小米空调不好": {
        "description": "小米品牌空调产品的性价比、制冷效果与售后评价争议。",
        "favor_reason": "质疑制冷制热效果、噪音、安装售后或代工厂品质不稳定。",
        "against_reason": "认可价格亲民、米家生态联动或外观设计简洁。",
    },
    "淘宝天猫预售制度不好": {
        "description": "电商平台预售定金、发货周期与退订规则的消费者争议。",
        "favor_reason": "批评发货拖延、退定金困难、规则复杂或绑架消费者。",
        "against_reason": "认为预售锁定优惠、分摊商家备货压力或便于大促组织。",
    },
    "国产手机不好": {
        "description": "中国本土手机品牌与产品的综合消费评价争议。",
        "favor_reason": "认为系统广告多、高端形象不足、芯片依赖或品控参差。",
        "against_reason": "认可性价比高、功能迭代快、本土化体验或支持国货。",
    },
    "不应该消除对罪犯子女考公的限制": {
        "description": "是否应维持罪犯直系亲属报考公务员政审限制的争论。",
        "favor_reason": "认为政审必要、防范利益输送或维护公职队伍纯洁性。",
        "against_reason": "认为连坐违背法治、子女不应代父受过或限制机会平等。",
    },
    "爱彼迎不好": {
        "description": "Airbnb短租平台及民宿模式的便利性与社区影响评价。",
        "favor_reason": "批评扰民、治安难管、逃税或冲击小区秩序与正规酒店。",
        "against_reason": "认可价格灵活、体验本地化或给房东增加收入渠道。",
    },
    "反对百度无人驾驶“萝卜快跑”": {
        "description": "百度Robotaxi在武汉等地商业化是否应被限制或欢迎的争议。",
        "favor_reason": "担心抢夺网约车司机饭碗、事故责任不清或技术未成熟。",
        "against_reason": "认可技术示范、低价出行或推动智能交通与产业升级。",
    },
    "反对iPhone16取消实体音量键电源键": {
        "description": "苹果取消实体按键改用触控或固态按键的设计争议。",
        "favor_reason": "批评盲操不便、误触风险、维修成本或牺牲实用性换外观。",
        "against_reason": "认可一体化设计、防水防尘改进或新交互更简洁。",
    },
    "茶颜悦色不好": {
        "description": "长沙茶饮品牌茶颜悦色的口味、排队与品牌溢价评价争议。",
        "favor_reason": "认为名不副实、排队营销过度、价格偏高或外地扩张变味。",
        "against_reason": "认可茶香口感、国风包装或认为值得排队体验。",
    },
    "反对补课": {
        "description": "学生课外补习是否应被禁止或严格限制的公共讨论。",
        "favor_reason": "支持双减、认为补习加剧内卷、掏空家庭或违背减负初衷。",
        "against_reason": "认为补习能查漏补缺、满足升学竞争下的现实需求。",
    },
    "360杀毒软件不好": {
        "description": "360安全卫士及杀毒软件的用户体验与商业模式评价。",
        "favor_reason": "批评捆绑安装、弹窗骚扰、占用资源或隐私收集问题。",
        "against_reason": "认可免费防护、木马查杀能力或对普通用户够用。",
    },
    "反对孩子上少儿编程课": {
        "description": "面向少儿的编程培训是否必要、是否贩卖焦虑的争议。",
        "favor_reason": "质疑机构营销焦虑、课程质量参差或与低龄认知不匹配。",
        "against_reason": "认为培养逻辑思维、适应未来科技社会或兴趣拓展有益。",
    },
    "特斯拉不好": {
        "description": "特斯拉品牌电动汽车的质量、安全与服务的公众评价争议。",
        "favor_reason": "批评刹车门、做工粗糙、售后差或马斯克言论引发反感。",
        "against_reason": "认可续航智能化、品牌力或认为问题被舆论放大。",
    },
    "家长不应该对孩子进行打骂教育": {
        "description": "家长体罚、打骂孩子是否属于可接受管教方式的争论。",
        "favor_reason": "认为打骂伤害身心健康、属家庭暴力、应依法保护儿童。",
        "against_reason": "认为适度惩戒有效、传统管教被过度政治正确或难以替代。",
    },
    "反对盲盒文化": {
        "description": "盲盒抽卡式消费是否诱导未成年人、应否被限制的讨论。",
        "favor_reason": "认为类似赌博、诱导非理性消费或坑害青少年。",
        "against_reason": "认为属正常娱乐消费、收藏乐趣或应靠监管而非一概禁止。",
    },
    "民宿不应该开在小区里": {
        "description": "住宅小区内经营短租民宿是否侵犯业主权益的争议。",
        "favor_reason": "批评陌生人流动、扰民治安风险或占用公共资源。",
        "against_reason": "认可盘活闲置房源、合法增收或认为可规范而非禁止。",
    },
    "反对取消公办中小学教师编制": {
        "description": "是否应保留公办教师事业编制、反对合同制改革的争论。",
        "favor_reason": "担心降低职业吸引力、待遇不稳或削弱教育公益性保障。",
        "against_reason": "认为打破铁饭碗能优胜劣汰、提高教学积极性与灵活性。",
    },
    "空气炸锅不好": {
        "description": "空气炸锅作为厨房小家电的健康概念与实用性评价。",
        "favor_reason": "认为口感不如油炸、容量小难清洗或健康效果被夸大。",
        "against_reason": "认可少油烹饪、操作简便或适合懒人快手菜。",
    },
    "反对男性化妆": {
        "description": "男性使用化妆品与护肤是否违背传统男子气概的争议。",
        "favor_reason": "认为不符合传统男性气质、过度消费或影响青少年审美。",
        "against_reason": "认为个人形象管理自由、打破性别刻板印象。",
    },
    "中学生不应该做头发": {
        "description": "中学生染发烫发是否应被校规禁止的公共讨论。",
        "favor_reason": "支持校规限制、认为学生应朴素专注或担心攀比风气。",
        "against_reason": "认为发型是个人自由、适度打扮不影响学习。",
    },
    "反对人大代表建议把体育由副科变主科": {
        "description": "是否应拒绝将体育升格为主科、维持副科地位的争论。",
        "favor_reason": "认为加重负担、师资场地不足、主科已饱和且易流于形式。",
        "against_reason": "认为可增强青少年体质、落实锻炼要求、纠正重文轻体。",
    },
    "上海地铁不应有板凳族": {
        "description": "上海地铁乘客自带小板凳乘车是否应被禁止的管理争议。",
        "favor_reason": "认为占通道碍事、增加安全隐患或应遵守禁带规定。",
        "against_reason": "理解通勤距离长、认为自备座位是无奈之举。",
    },
    "中小学应允许校内设置小卖部超市": {
        "description": "校园内是否应恢复或允许设置小卖部、超市的争议。",
        "favor_reason": "认为给学生提供便利、满足课间消费需求。",
        "against_reason": "认为高盐高糖零食危害健康、应统一供餐而非校内售卖。",
    },
    "女孩不应穿露背装在有轨电车拍照": {
        "description": "女子着露背装在公共交通工具上拍照的公共礼仪争议。",
        "favor_reason": "认为在公共交通过于暴露、影响他人或缺乏场合分寸。",
        "against_reason": "认为穿衣自由、未违法且不应被道德绑架。",
    },
    "反对代孕": {
        "description": "代孕行为是否应被法律禁止、如何界定伦理边界的争论。",
        "favor_reason": "强调代孕违法、剥削女性身体或带来伦理与买卖风险。",
        "against_reason": "认为可帮助不孕不育家庭、应适度合法规范。",
    },
    "反对萝卜快跑无人驾驶出租车": {
        "description": "百度萝卜快跑Robotaxi商业化对出行行业与就业的冲击争议。",
        "favor_reason": "担心失业问题、安全责任、道路适应不足或垄断风险。",
        "against_reason": "认可出行新模式、低价体验或技术产业进步。",
    },
    "反对燃放烟花爆竹": {
        "description": "春节等传统节日燃放烟花爆竹是否应禁放限放的争论。",
        "favor_reason": "强调空气污染、火灾伤人风险或支持全域禁放。",
        "against_reason": "认为体现年俗文化、增加节日气氛或应适度放开。",
    },
    "反对取消公务员考试35岁限制": {
        "description": "公务员招录是否应维持35周岁年龄上限的公共讨论。",
        "favor_reason": "认为保障队伍年轻化、体能要求或与岗位特性匹配。",
        "against_reason": "认为构成年龄歧视、浪费中年人才或应放宽上限。",
    },
    "反对学校设置课后服务": {
        "description": "中小学课后延时服务政策是否应被叫停或改革的争议。",
        "favor_reason": "批评变相延长在校时间、增加师生负担或流于形式。",
        "against_reason": "认为解决家长接送难题、提供看护与辅导资源。",
    },
    "反对燃油车": {
        "description": "传统燃油汽车是否应被加速淘汰、全面转向新能源的争论。",
        "favor_reason": "强调排放污染、能源依赖或认为应加速电动化。",
        "against_reason": "认可加油方便、技术成熟、长途可靠或无里程焦虑。",
    },
    "支持对罪犯子女考公限制": {
        "description": "是否应支持对罪犯直系亲属报考公务员实施政审限制。",
        "favor_reason": "认为政审必要、防范利益输送或维护公职队伍纯洁性。",
        "against_reason": "认为连坐违背法治、子女不应代父受过或限制机会平等。",
    },
    "反对直播带货": {
        "description": "直播电商模式是否应被限制、其利弊的公共评价。",
        "favor_reason": "批评虚假宣传、退货难、冲动消费或头部主播垄断流量。",
        "against_reason": "认可低价优惠、直观展示或帮助农产品与中小商家。",
    },
    "支持结婚不用征得父母同意": {
        "description": "婚姻是否无需父母同意、强调个人自主权的观点争议。",
        "favor_reason": "主张婚姻自主、反对父母包办或认为成年人有权独立决定。",
        "against_reason": "认为父母经验可把关、尊重家庭伦理或避免冲动结合。",
    },
    "反对流浪狗": {
        "description": "城市流浪犬治理方式、是否应捕杀或人道救助的争议。",
        "favor_reason": "强调公共安全、卫生隐患或支持严格捕杀与管理。",
        "against_reason": "主张人道救助、领养绝育或认为应宽容共存。",
    },
    "家长不应对孩子进行打骂教育": {
        "description": "家长是否不应对孩子实施打骂式管教的规范讨论。",
        "favor_reason": "认为打骂伤害身心健康、属家庭暴力、应依法保护儿童。",
        "against_reason": "认为适度惩戒有效、传统管教被过度政治正确或难以替代。",
    },
    "反对电车": {
        "description": "是否应反对购买或使用电动汽车的公众态度讨论。",
        "favor_reason": "质疑续航焦虑、电池衰减、充电排队或冬季性能下降。",
        "against_reason": "认可使用成本低、加速平顺、智能化或环保趋势。",
    },
    "认为超雄基因宝宝可以被生下来": {
        "description": "XYY等超雄胎儿是否应被允许出生、反对选择性终止的伦理讨论。",
        "favor_reason": "强调基因不等于行为、反对标签化堕胎、生命权应受保护。",
        "against_reason": "认为应尊重知情选择、减轻家庭抚养风险或基于医学建议终止。",
    },
    "反对防止男性青少年女性化提案": {
        "description": "是否应反对「防止男生女性化」类教育提案的公共争论。",
        "favor_reason": "批评定义模糊、歧视多元表达或把气质问题简单政治化。",
        "against_reason": "认为应加强体育与意志锻炼、纠正过度阴柔审美。",
    },
    "中小学教师不应该到培训机构兼职": {
        "description": "公办教师在校外培训机构兼职是否应被禁止的争议。",
        "favor_reason": "担心课上不教课下教、加剧教育不公或违反禁补规定。",
        "against_reason": "认为教师可利用业余时间合法增收、发挥专业特长。",
    },
    "反对深圳中小学延后两小时放学": {
        "description": "深圳推迟中小学放学时间以衔接家长下班的政策争议。",
        "favor_reason": "担心学生疲劳、挤占休息或加重教师与家庭调度压力。",
        "against_reason": "认为匹配家长下班时间、减少托管真空。",
    },
    "反对禁止单身女性冷冻卵子提案": {
        "description": "是否应反对限制未婚女性冻卵的政策提案的争论。",
        "favor_reason": "主张女性应有冻卵自主权、不应因婚姻状况被剥夺。",
        "against_reason": "认为应维护现行生育法规秩序、防止辅助生殖滥用。",
    },
    "反对冻卵": {
        "description": "女性冷冻卵子技术是否应被限制或普遍推广的伦理争议。",
        "favor_reason": "担心伦理风险、商业炒作或认为应优先自然生育。",
        "against_reason": "认为保留生育选择权、帮助晚育或职业女性规划。",
    },
    "癌症晚期没必要倾家荡产吃靶向药延长生命": {
        "description": "晚期癌症是否值得倾家荡产使用靶向药延命的家庭伦理讨论。",
        "favor_reason": "认为应理性止损、保障家庭生存质量或接受生命终点。",
        "against_reason": "认为尽力救治是亲情责任、新药可能延长有质量生命。",
    },
    "反对网络歌曲": {
        "description": "网络神曲、短视频BGM等是否拉低审美、应被抵制的争论。",
        "favor_reason": "认为歌词低俗、旋律洗脑、冲击传统音乐或青少年审美。",
        "against_reason": "认为属流行文化正常现象、娱乐无罪或不应道德审判。",
    },
    "认为教师工作不轻松": {
        "description": "反驳「教师工作轻松」说法、强调职业强度的公共讨论。",
        "favor_reason": "强调备课批改、班主任、迎检与低薪，认为并不轻松。",
        "against_reason": "认为寒暑假长、工作稳定、相比企业压力较小。",
    },
    "反对孕妇未做胸透被拒录用": {
        "description": "孕妇因未做胸透被拒录用是否构成歧视的就业争议。",
        "favor_reason": "认为构成性别歧视、应提供替代检查或保护孕妇就业权。",
        "against_reason": "认为单位需履行体检合规、岗位对健康有合理要求。",
    },
    "反对赋予单身女性实施辅助生育技术权利": {
        "description": "未婚女性是否应享有辅助生殖权利的公共争论。",
        "favor_reason": "认为冲击传统家庭伦理、应优先婚内生育或防止滥用。",
        "against_reason": "支持女性生育自主权、认为婚姻状况不应限制医疗权利。",
    },
    "不应该禁止男童进女厕": {
        "description": "是否不应禁止母亲带年幼男孩进入女卫生间的争议。",
        "favor_reason": "认为单亲妈妈带娃无处可去、应设亲子厕所而非简单禁止。",
        "against_reason": "认为应保障女性如厕隐私、减少异性儿童进入。",
    },
    "反对新冠躺平论": {
        "description": "是否应反对「与病毒共存、全面放开」的新冠防控主张。",
        "favor_reason": "强调医疗挤兑风险、老人脆弱群体保护或认为放开过快。",
        "against_reason": "认为应与病毒共存、恢复经济正常生活、过度防控代价太大。",
    },
    "反对顾客取消外卖订单": {
        "description": "消费者是否不应随意取消外卖订单、损害骑手商家利益的争论。",
        "favor_reason": "认为临近出餐取消伤害骑手商家、应扣费或限制滥用。",
        "against_reason": "认为消费者有权反悔、平台应保障用户灵活取消。",
    },
    "不应该消除简历第一学历概念": {
        "description": "招聘是否不应弱化第一学历、仍应看重本科出身的争议。",
        "favor_reason": "认为第一学历仍有区分度、一刀切可能降低筛选效率。",
        "against_reason": "认为给专升本、考研者更多机会、减少学历歧视。",
    },
    "反对上补课班": {
        "description": "是否应反对学生参加校外学科类补课班的公共讨论。",
        "favor_reason": "支持双减、认为非法补课加剧不公平与学生负担。",
        "against_reason": "认为补习必要、可针对性提分或家长自愿选择。",
    },
    "海底捞不好": {
        "description": "连锁火锅品牌海底捞的服务、价格与口味评价争议。",
        "favor_reason": "认为价格偏高、口味一般、服务表演化或性价比下降。",
        "against_reason": "认可服务态度、就餐体验或家庭聚餐选择。",
    },
    "妈妈不应该带男童进女厕": {
        "description": "母亲是否不应带年幼男孩进入女卫生间的公共讨论。",
        "favor_reason": "认为侵犯女性隐私、男孩应有年龄上限或需第三卫生间。",
        "against_reason": "认为幼儿需要照顾、母亲别无选择应被理解。",
    },
    "反对丰巢快递柜": {
        "description": "丰巢智能快递柜超时收费与小区占用空间的争议。",
        "favor_reason": "批评超时收费不合理、强占公共空间或未征得业主同意。",
        "against_reason": "认可解决收件时间冲突、提高投递效率。",
    },
    "反对家庭教育不得有任何形式家庭暴力": {
        "description": "是否应反对法律明确禁止家庭暴力的规范条款。",
        "favor_reason": "担心定义过宽、干预正常管教或认为应留给家庭更多空间。",
        "against_reason": "认为应立法保护儿童、明确打骂亦属家暴、推动科学育儿。",
    },
    "反对打新冠疫苗": {
        "description": "是否应反对接种新冠疫苗的公众态度争议。",
        "favor_reason": "担心副作用、认为个人可自主选择或质疑强制接种。",
        "against_reason": "认为应积极接种、降低重症风险、履行公共健康责任。",
    },
    "反对酒局文化": {
        "description": "职场酒桌劝酒、拼酒习俗是否应被抵制改革的争论。",
        "favor_reason": "批评强迫饮酒、健康损害、权力压迫或性别不平等。",
        "against_reason": "认为酒桌是沟通感情、商务礼仪的传统方式。",
    },
    "学文科不好": {
        "description": "选择文科专业是否不明智、就业与收入是否偏低的讨论。",
        "favor_reason": "认为就业面窄、收入偏低或不如理工科实用。",
        "against_reason": "认为培养人文素养、批判思维或社会理解力不可替代。",
    },
    "反对像国外一样取消防疫措施": {
        "description": "是否应反对参照国外全面放开、取消严格新冠防控。",
        "favor_reason": "强调医疗挤兑风险、老人等脆弱群体需保护、不宜照搬国外。",
        "against_reason": "认为应恢复经济正常生活、过度封控代价大、与病毒共存。",
    },
    "反对降低英语教学比重建议": {
        "description": "是否应反对减少英语在中小学课程中比重的改革建议。",
        "favor_reason": "认为英语仍重要、国际交流需要或降低比重削弱竞争力。",
        "against_reason": "认为减轻学生负担、强化母语或英语实用性被高估。",
    },
    "地铁不应该单独设置女性车厢": {
        "description": "地铁是否不应设女性专用车厢、反对性别隔离的争论。",
        "favor_reason": "认为强化性别对立、挤占公共资源或男性也需安全空间。",
        "against_reason": "认为可防猥亵、提升女性通勤安全感。",
    },
    "认为学区房未来会贬值": {
        "description": "学区房价是否将因教育均衡政策而下跌的预期讨论。",
        "favor_reason": "认为多校划片、教师轮岗等政策将削弱学区溢价。",
        "against_reason": "认为优质教育资源仍稀缺、家长择校需求长期存在。",
    },
    "反对老师比公务员累": {
        "description": "是否应反对「教师比公务员更累」这一职业比较说法。",
        "favor_reason": "认为公务员同样加班迎检、不应单方面强调教师更累。",
        "against_reason": "强调班主任、课后服务、家校沟通等使教师负担更重。",
    },
    "反对丁克": {
        "description": "选择不生育的丁克生活方式是否应被批评的公共争论。",
        "favor_reason": "认为违背传宗接代、加剧老龄化或自私不负责任。",
        "against_reason": "认为生育是个人自由、不应道德绑架或强迫生育。",
    },
    "反对学校课后服务": {
        "description": "中小学课后延时服务是否应被取消或缩减的争议。",
        "favor_reason": "批评变相延长在校时间、增加师生负担或流于形式。",
        "against_reason": "认为解决家长接送难题、提供看护与辅导资源。",
    },
    "反对有兄弟姐妹赡养老人压力更小的观点": {
        "description": "是否应反对「多子女分担养老更轻松」的传统观念。",
        "favor_reason": "认为多子女可能推诿责任、内耗更多或养老不应依赖子女数量。",
        "against_reason": "认为兄弟姐妹可轮流照顾、经济与时间压力可分担。",
    },
    "教师工作中不应存在各种和教学无关的额外职责和压力": {
        "description": "教师是否应免于迎检、填表、行政等非教学任务的争论。",
        "favor_reason": "认为应减负、让教师专注教学、反对形式主义摊派。",
        "against_reason": "认为学校管理需要、家校社协同或部分事务难以剥离。",
    },
    "反对新能源电车": {
        "description": "是否应普遍反对新能源电动汽车推广与购买的争论。",
        "favor_reason": "质疑充电不便、电池安全、保值率或认为现阶段不如燃油车。",
        "against_reason": "认可低碳出行、使用成本低、智能化体验或产业自主发展。",
    },
    "光明日报刊文“娘炮形象”等畸形审美必须遏制": {
        "description": "官媒批评「娘炮」等男性阴柔审美是否正当、如何界定畸形。",
        "favor_reason": "认为需引导阳刚之气、抵制过度阴柔化对青少年影响。",
        "against_reason": "认为审美应多元、用词污名化或不应由媒体定义男性气质。",
    },
    "教师招聘中男性更有优势": {
        "description": "中小学教师招聘是否存在隐性偏好男性、性别比例失衡的讨论。",
        "favor_reason": "认为男教师稀缺、需平衡性别或男生需要男性榜样。",
        "against_reason": "认为应唯能力论、反对性别配额或女性同样胜任。",
    },
    "以成绩为评价学生的唯一标准的教育制度存在问题": {
        "description": "唯分数论、应试教育是否损害学生全面发展的公共批评。",
        "favor_reason": "认为忽视素质、心理健康与创新能力、加剧内卷。",
        "against_reason": "认为分数相对公平、可操作或改革需渐进不能全盘否定。",
    },
    "生二胎要征求大孩子的意见": {
        "description": "生育二孩是否应征得头胎子女同意的家庭伦理讨论。",
        "favor_reason": "认为尊重孩子感受、减少家庭矛盾或保障其心理安全。",
        "against_reason": "认为生育权在父母、不应让儿童决定或过度赋权。",
    },
    "应当由家长检查学生课后作业": {
        "description": "课后作业是否应由家长负责检查、而非完全交给学校的争论。",
        "favor_reason": "认为家长应参与学习、及时发现错误或培养习惯。",
        "against_reason": "认为增加家长负担、专业批改应在学校或双职工难以做到。",
    },
    "反对江歌妈妈": {
        "description": "江歌案受害者母亲刘鑫（江歌妈妈）相关舆论与网络评价争议。",
        "favor_reason": "认为其维权过度、消费悲剧或网络募捐透明度存疑。",
        "against_reason": "同情丧女之痛、认可其推动司法与公共讨论。",
    },
    "抵制病态整容娘炮审美": {
        "description": "是否应抵制过度整容、阴柔「娘炮」类审美风向的公共讨论。",
        "favor_reason": "认为危害青少年价值观、导向畸形消费或丧失阳刚之气。",
        "against_reason": "认为审美应多元、反对污名化或不应由公权力定义。",
    },
    "体制内工作不好": {
        "description": "公务员、事业单位等体制内岗位是否值得选择的评价争议。",
        "favor_reason": "认为晋升慢、工资低、形式主义或缺乏成就感。",
        "against_reason": "认可稳定、福利、社会声望或工作生活平衡。",
    },
    "反对取消新冠防控全面放开": {
        "description": "是否应反对调整防控、全面放开的新冠政策转向。",
        "favor_reason": "强调医疗挤兑风险、老人脆弱群体保护或放开节奏过快。",
        "against_reason": "认为应恢复经济正常生活、严格管控代价过大。",
    },
    "春节不应禁放烟花": {
        "description": "春节是否不应全面禁放烟花爆竹、应保留年俗的讨论。",
        "favor_reason": "认为体现年俗文化、增加节日气氛或应适度放开。",
        "against_reason": "强调空气污染、火灾伤人风险或支持全域禁放。",
    },
    "分手后一定要删除对方": {
        "description": "恋爱结束后是否必须删除前任联系方式的社交观念争议。",
        "favor_reason": "认为有助于断联疗愈、避免纠缠或尊重新恋情。",
        "against_reason": "认为可和平共处、保留友谊或删除并非必要。",
    },
    "反对中国人在南京穿和服": {
        "description": "南京等特殊历史语境下国人着和服是否应被反对的争论。",
        "favor_reason": "认为伤害民族感情、忽视历史伤痛或场合极不恰当。",
        "against_reason": "认为穿衣自由、不应将服饰与政治简单挂钩。",
    },
    "梅西提前下场应赔偿违约金": {
        "description": "梅西香港表演赛未出场是否应担违约责任、赔偿球迷的讨论。",
        "favor_reason": "认为欺骗球迷、商业契约应履约或损害中国球迷感情。",
        "against_reason": "认为伤病免责、合同条款复杂或不应简单要求赔偿。",
    },
    "反对不育主义": {
        "description": "主动选择不生育的不育主义是否应被社会批评的争论。",
        "favor_reason": "认为违背人口发展、加剧老龄化或不负责任。",
        "against_reason": "认为生育是个人自由、不应道德绑架。",
    },
    "排斥同性恋": {
        "description": "对同性恋群体持排斥、否定态度是否合理的公共讨论。",
        "favor_reason": "认为违背传统家庭伦理、不应公开宣扬或担心对青少年影响。",
        "against_reason": "主张性取向平等、反对污名化或认为属个人自由与基本权利。",
    },
    "董明珠称：“大学生去打螺钉没什么不可以，聪明人应该走基层”": {
        "description": "董明珠关于大学生应下基层、打螺钉言论引发的争议。",
        "favor_reason": "认为应脚踏实地、基层锻炼有益或反对眼高手低。",
        "against_reason": "认为矮化学历价值、与岗位匹配无关或话术忽视人才结构。",
    },
    "反对带男童进女厕": {
        "description": "是否应反对母亲带年幼男孩进入女卫生间的公共讨论。",
        "favor_reason": "认为应保障女性如厕隐私、减少异性儿童进入。",
        "against_reason": "认为单亲妈妈带娃无处可去、应设亲子厕所而非简单禁止。",
    },
    "支持公务员考试35岁限制": {
        "description": "是否应支持公务员招录维持35周岁年龄上限。",
        "favor_reason": "认为保障队伍年轻化、体能要求或与岗位特性匹配。",
        "against_reason": "认为构成年龄歧视、浪费中年人才或应放宽上限。",
    },
    "女性不应穿着暴露这一观点": {
        "description": "女性着装是否不应过于暴露、公共场合穿衣尺度的争论。",
        "favor_reason": "认为过于暴露不合公序良俗、易引骚扰或应保守得体。",
        "against_reason": "认为穿衣自由、反对 victim blaming 或不应规训女性身体。",
    },
    "应保持对罪犯子女考公限制": {
        "description": "是否应维持罪犯直系亲属报考公务员的政审限制。",
        "favor_reason": "认为政审必要、防范利益输送或维护公职队伍纯洁性。",
        "against_reason": "认为连坐违背法治、子女不应代父受过或限制机会平等。",
    },
    "俞敏洪说在中国，如果完全没有人脉，一切凭着公事公办的方式想把事情做成，难度是比较大的": {
        "description": "俞敏洪关于中国人脉社会、公事公办难成事的言论争议。",
        "favor_reason": "认为反映现实、关系网普遍或点出制度外潜规则。",
        "against_reason": "认为贩卖焦虑、以偏概全或应推动规则透明而非认命。",
    },
    "应同情提供代孕服务的印度女性": {
        "description": "印度代孕产业中底层女性是否值得同情、如何看跨境代孕。",
        "favor_reason": "认为她们被剥削、贫困被迫出租子宫、人权应受关注。",
        "against_reason": "认为自愿交易、改善生计或问题在监管而非一概同情。",
    },
    "广电总局说坚决抵制病态整容娘炮审美": {
        "description": "广电总局抵制「娘炮」等畸形审美表态的公众支持与质疑。",
        "favor_reason": "认为需净化荧屏、引导青少年健康审美与阳刚之气。",
        "against_reason": "认为审美应多元、用语污名化或干预过度。",
    },
    "反对彩礼习俗": {
        "description": "传统婚嫁彩礼习俗是否应被废除或严格限制的争论。",
        "favor_reason": "批评高额彩礼加重负担、物化婚姻或变相买卖。",
        "against_reason": "认为彩礼体现诚意、尊重女方家庭或符合地方婚俗传统。",
    },
    "厌恶同性恋": {
        "description": "对同性恋持厌恶、排斥情绪是否合理的公共态度讨论。",
        "favor_reason": "认为违背传统家庭伦理、不应公开宣扬或担心对青少年影响。",
        "against_reason": "主张性取向平等、反对污名化或认为属个人自由与基本权利。",
    },
    "排斥彩礼": {
        "description": "对婚嫁彩礼习俗持排斥、否定态度的公共讨论。",
        "favor_reason": "批评高额彩礼加重负担、物化婚姻或变相买卖。",
        "against_reason": "认为彩礼体现诚意、尊重女方家庭或符合地方婚俗传统。",
    },
    "女子出嫁仍应该有村民资格有权分土地": {
        "description": "出嫁女是否仍应享有村集体成员资格与土地权益的争议。",
        "favor_reason": "认为男女平等、户口未迁不应剥夺权益或属性别歧视。",
        "against_reason": "认为传统村规、避免外嫁女双重受益或集体资源有限。",
    },
    "关停下架拍照搜题APP": {
        "description": "是否应关停或下架学生拍照搜题类作业App的政策讨论。",
        "favor_reason": "认为助长抄答案、削弱独立思考、违背作业训练目的。",
        "against_reason": "认为可辅助解题、减轻家长辅导压力或应规范而非关停。",
    },
    "排斥打新冠疫苗": {
        "description": "对接种新冠疫苗持排斥、拒绝态度的公众讨论。",
        "favor_reason": "担心副作用、认为个人可自主选择或质疑强制接种。",
        "against_reason": "认为应积极接种、降低重症风险、履行公共健康责任。",
    },
    "马斯克认为不应帮助人类长寿": {
        "description": "马斯克反对过度延长人类寿命、认为应接受死亡的观点争议。",
        "favor_reason": "认为资源应优先解决当下问题、长寿加剧不平等或地球承载力。",
        "against_reason": "认为延长健康寿命是医学进步、个人有权追求或言论过于极端。",
    },
    "反对按性别分配软卧车厢": {
        "description": "火车软卧是否不应按性别分配铺位、反对混住限制的争论。",
        "favor_reason": "认为侵犯隐私、女性安全需求或应提供可选而非强制分性别。",
        "against_reason": "认为混住易生纠纷、女性乘客更安心或运营方有权管理。",
    },
    "互联网平台账号不应显示IP属地": {
        "description": "微博等平台显示用户IP属地是否应被取消的功能争议。",
        "favor_reason": "担心暴露隐私、属地不代表真实身份或流于形式。",
        "against_reason": "认为有助于识别造谣账号、减少冒充身份与网络暴力。",
    },
    "质疑姜萍": {
        "description": "中专生姜萍阿里数学竞赛成绩是否真实、是否存在作弊的舆论争议。",
        "favor_reason": "质疑作弊可能、老师辅助过度或认为叙事被夸大造神。",
        "against_reason": "相信成绩真实、认可天赋与努力打破学历偏见。",
    },
    "王楚钦输球不是因为换拍子": {
        "description": "王楚钦巴黎奥运换拍后输球是否应归咎于球拍的讨论。",
        "favor_reason": "认为应尊重裁判装备规则、输球主因是竞技状态而非器材。",
        "against_reason": "认为备用拍差异大、影响发挥或组委会准备不足。",
    },
    "不愿意打新冠疫苗": {
        "description": "个人拒绝或犹豫接种新冠疫苗的态度与理由争议。",
        "favor_reason": "担心副作用、认为个人可自主选择或质疑保护效果。",
        "against_reason": "认为应积极接种、降低重症风险、履行公共健康责任。",
    },
    "反对西安某高校设置禁酒令": {
        "description": "高校校园全面禁酒规定是否过度、应否被反对的争论。",
        "favor_reason": "认为大学生有饮酒自由、禁酒侵犯权利或难以执行。",
        "against_reason": "认为维护校园秩序、减少酗酒滋事或符合校规管理需要。",
    },
    "酒店不允许成年子女和父母住一个标间": {
        "description": "酒店拒绝成年子女与父母同住标间是否合理的消费争议。",
        "favor_reason": "认为涉嫌年龄歧视、家庭出行不便或应灵活安排。",
        "against_reason": "认为标间限住人数、消防与安全规定或避免变相多人入住。",
    },
    "超雄基因孩子应被打掉": {
        "description": "发现超雄（XYY）等胎儿异常时是否应终止妊娠的伦理争议。",
        "favor_reason": "认为应尊重知情选择、减轻家庭抚养风险或基于医学建议终止。",
        "against_reason": "强调基因不等于行为、反对标签化堕胎、生命权应受保护。",
    },
}


def has_template_marker(entry: dict) -> bool:
    text = " ".join(
        str(entry.get(k, "")) for k in ("description", "favor_reason", "against_reason")
    )
    return any(marker in text for marker in TEMPLATE_MARKERS)


def main() -> None:
    with TEMPLATE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    bad_claims = json.loads(BAD_CLAIMS_PATH.read_text(encoding="utf-8"))
    if len(bad_claims) != 132:
        raise SystemExit(f"Expected 132 bad claims, got {len(bad_claims)}")
    if len(UPDATES) != 132:
        raise SystemExit(f"Expected 132 UPDATES entries, got {len(UPDATES)}")

    missing_updates = [t for t in bad_claims if t not in UPDATES]
    if missing_updates:
        raise SystemExit(f"Missing UPDATES for: {missing_updates}")

    extra_updates = [t for t in UPDATES if t not in bad_claims]
    if extra_updates:
        raise SystemExit(f"Extra UPDATES not in bad claims: {extra_updates}")

    missing_targets = [t for t in bad_claims if t not in data]
    if missing_targets:
        raise SystemExit(f"Missing targets in template: {missing_targets}")

    patched = 0
    for target in bad_claims:
        entry = data[target]
        update = UPDATES[target]
        entry["description"] = update["description"]
        entry["favor_reason"] = update["favor_reason"]
        entry["against_reason"] = update["against_reason"]
        patched += 1

    for target, entry in data.items():
        if target in UPDATES:
            continue
        if not has_template_marker(entry):
            continue
        raise SystemExit(
            f"Entry '{target}' matches template markers but has no UPDATE"
        )

    neutral_removed = 0
    for entry in data.values():
        if "neutral_hint" in entry:
            del entry["neutral_hint"]
            neutral_removed += 1

    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    remaining_markers = sum(1 for v in data.values() if has_template_marker(v))
    remaining_neutral = sum(1 for v in data.values() if "neutral_hint" in v)

    print(f"Patched claim entries: {patched}")
    print(f"Removed neutral_hint from: {neutral_removed} entries")
    print(f"Total entries: {len(data)}")
    print(f"Remaining template markers: {remaining_markers}")
    print(f"Remaining neutral_hint keys: {remaining_neutral}")

    if remaining_markers or remaining_neutral:
        raise SystemExit("Verification failed")

    samples = ["人大代表建议把体育由副科变主科", "超雄基因宝宝不应该生下来", "教师工作轻松"]
    print("\nSample rewritten entries:")
    for name in samples:
        e = data[name]
        print(f"\n[{name}]")
        print(f"  description: {e['description']}")
        print(f"  favor_reason: {e['favor_reason']}")
        print(f"  against_reason: {e['against_reason']}")
        print(f"  target_type: {e['target_type']}")


if __name__ == "__main__":
    main()
