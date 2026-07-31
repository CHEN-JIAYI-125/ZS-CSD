#!/usr/bin/env python3
"""Refresh templated dislike / 反感 / 不喜欢 claim entries in target_knowledge_template.json."""

import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "data" / "target_knowledge_template.json"

TEMPLATE_MARKERS = (
    "情绪性命题",
    "赞同讨厌",
    "赞同不喜欢",
    "赞同反感",
)

UPDATES = {
    "讨厌杜兰特": {
        "description": "对NBA球星杜兰特表达厌恶或强烈反感的公共态度，常与其转会抱团、关键球表现、球迷群体对立及「投敌」标签相关。",
        "favor_reason": "认同对其团队角色自私、关键场次不稳、场外言行或「抱团夺冠」路径的批评，认为这种讨厌有竞技与情感依据。",
        "against_reason": "认为应就球论球、反对人身攻击与饭圈化仇恨，其得分能力与职业选择不应被情绪性否定。",
    },
    "讨厌带货主播": {
        "description": "对直播带货主播群体表达反感的社会情绪，涉及虚假宣传、头部垄断流量、冲动消费与网红经济对实体店的挤压。",
        "favor_reason": "认为话术诱导下单、控价压小商家、售后维权难，讨厌反映对低质流量经济和消费主义的不满。",
        "against_reason": "认为带货降低信息成本、助农与中小品牌触达用户，不应因个别乱象否定职业本身。",
    },
    "讨厌王一博": {
        "description": "对演员、偶像王一博表达厌恶的网络舆论，常围绕演技争议、流量占资源、粉丝控评引战及商业代言刷屏。",
        "favor_reason": "认为其演技与顶流地位不匹配、粉丝行为引战或过度营销制造信息污染，讨厌情有可原。",
        "against_reason": "认为应区分艺人本人与饭圈，其舞台努力与商业价值有客观基础，不应被标签化攻击。",
    },
    "讨厌孙楠": {
        "description": "对歌手孙楠表达强烈反感的公众情绪，多与《歌手》退赛风波、倚老卖老印象、赛制尊重及近年表现落差相关。",
        "favor_reason": "认为退赛不尊重观众与比赛规则、态度傲慢或名望与现场表现不匹配，讨厌来自失信与落差。",
        "against_reason": "认为应尊重个人健康与选择，经典作品与早年贡献不应被一次事件盖过，反对网络泄愤式攻击。",
    },
    "讨厌流浪狗": {
        "description": "对城市流浪犬表达厌恶或希望严格管控的态度，常由咬人伤人事件、卫生隐患、夜间扰民与投喂争议触发。",
        "favor_reason": "强调犬只失控伤人、粪便污染、吠叫扰民等公共安全与卫生诉求，认为讨厌源于真实生活风险。",
        "against_reason": "认为流浪动物本身无罪，应人道救助、绝育领养而非情绪性厌恶，反对虐待或无差别清除。",
    },
    "讨厌石楠花": {
        "description": "对城市绿化石楠树开花气味表达厌恶的民生讨论，春季「臭花」嗅觉体验与生态效益、养护成本之间的冲突。",
        "favor_reason": "抱怨气味类似鱼腥味影响出行、开窗与通勤，认为绿化选材忽视居民日常感受。",
        "against_reason": "认为石楠四季常绿、易养护、生态价值高，短期气味可忍受，不应因嗅觉偏好全盘否定。",
    },
    "讨厌伊藤美诚": {
        "description": "对日本乒乓球运动员伊藤美诚表达厌恶的舆论，常与中日竞技对抗、赛场言行、媒体渲染及民族情绪交织。",
        "favor_reason": "认为其部分言行或比赛风格引发不适，讨厌带有竞技对立与历史语境下的情绪宣泄。",
        "against_reason": "认为应尊重对手、区分竞技与仇恨，其训练投入与水平值得客观评价，不应上升为人身攻击。",
    },
    "讨厌梅西": {
        "description": "对足球运动员梅西表达厌恶的公众态度，香港表演赛未出场等事件显著推高了对其职业态度与商业信誉的负面评价。",
        "favor_reason": "认为其对中国球迷不够尊重、商业履约失当或「球王」人设与行为不符，讨厌反映被辜负感。",
        "against_reason": "认为伤病与合同细节可能被误读，历史成就与球技贡献不应被单次事件定义，反对宣泄式网暴。",
    },
    "讨厌豆瓣app": {
        "description": "对豆瓣社区App表达厌恶的用户情绪，涉及内容审查删帖、产品功能停滞、小组氛围恶化与饭圈入侵评分。",
        "favor_reason": "批评评分失真、管理混乱、审核过度或长期不更新，讨厌反映对社区质量下滑的失望。",
        "against_reason": "认为其仍是相对理性的书影音讨论平台，问题多来自外部环境与监管，不应被一概否定。",
    },
    "讨厌吴艳妮": {
        "description": "对跨栏运动员吴艳妮表达厌恶的舆论，聚焦其高调造型、营销风格、抢跑争议与「网红运动员」标签。",
        "favor_reason": "认为其过度注重话题性与外表、分散对成绩的关注，或抢跑等行为损害体育精神观感。",
        "against_reason": "认为运动员表达个性无可厚非，厌恶常含性别双标，成绩进步与赛场努力应被正视。",
    },
    "讨厌李佳琦": {
        "description": "对头部直播带货主播李佳琦表达强烈反感的公众情绪，涉及「哪里贵了」言论、定价话语权与流量寡头化。",
        "favor_reason": "认为其话术脱离普通人收入、控价压榨品牌商家，讨厌代表对头部主播权力与消费PUA的反噬。",
        "against_reason": "认为其曾以议价惠及消费者、选品有贡献，一次失言或争议不应否定整个带货模式与个人努力。",
    },
    "反感吴艳妮": {
        "description": "对跨栏运动员吴艳妮表达反感情绪的公共讨论，涉及个性展示、赛场营销、抢跑事件与运动员形象边界。",
        "favor_reason": "认为其高调造型与话题操作喧宾夺主，或抢跑、舆论姿态让人不适，反感来自竞技体育气质冲突。",
        "against_reason": "认为公众对女运动员外表过于苛刻，反感被放大为道德审判，应更多关注成绩与训练付出。",
    },
    "不喜欢吴艳妮": {
        "description": "对跨栏运动员吴艳妮表达不喜欢或偏见的舆论，常与个性风格、媒体曝光度及竞技成绩评价交织。",
        "favor_reason": "认为其营销过度、抢跑争议或风格张扬不符合心中「运动员」形象，不喜欢属审美与价值观差异。",
        "against_reason": "认为不喜欢不应演变为网暴，其成绩与努力有客观价值，公众应容忍多元化的运动员表达。",
    },
    "不喜欢王楚钦": {
        "description": "对乒乓球运动员王楚钦表达不喜欢或负面偏见的舆论，涉及大赛稳定性、排名争议、粉丝对立与舆论待遇。",
        "favor_reason": "认为其关键场失误与舆论热度不匹配、赛场形象或脾气引发不适，不喜欢反映竞技预期落差。",
        "against_reason": "认为年轻选手状态起伏正常，应看长期排名与贡献，不喜欢不应演变为人身攻击或网暴。",
    },
    "不喜欢补课班": {
        "description": "对校外学科类补课班表达不喜欢或抵触的态度，与双减政策、升学内卷、家庭负担及地下培训治理相关。",
        "favor_reason": "认为补课加重学生负担、掏空家庭、制造不公平竞争，不喜欢是对教育异化与焦虑经济的反抗。",
        "against_reason": "认为在升学压力下补习是部分家庭的无奈选择，不应简单否定自愿提分的现实需求。",
    },
    "反感超雄基因宝宝": {
        "description": "对XYY等被称「超雄」胎儿或相关生育议题表达本能反感的伦理讨论，涉及基因标签化、行为风险与残障污名。",
        "favor_reason": "认为异常核型可能带来行为与抚养风险，选择性终止是负责任选择，反感源于对未知与标签化风险的担忧。",
        "against_reason": "强调基因不等于犯罪或暴力，反对用「超雄」妖魔化胎儿，认为生命权与知情选择应受保护。",
    },
    "反感吃预制菜": {
        "description": "对食用预制菜（料理包、中央厨房复热成品）表达反感的消费观念争议，涉及餐厅明示、口感与食品安全信任。",
        "favor_reason": "认为预制菜口感差、添加剂担忧、餐厅隐瞒现做涉嫌欺骗，反感是对知情权与饮食体验的捍卫。",
        "against_reason": "认为标准化预制在合规前提下可降本控质，只要明示且安全，不应被一概污名为「垃圾食品」。",
    },
    "反感男性化妆": {
        "description": "对男性使用化妆品、护肤或精致装扮表达反感的性别观念争论，涉及阳刚气质、消费文化与青少年审美引导。",
        "favor_reason": "认为男性化妆违背传统性别角色、过度消费或向青少年传递阴柔化审美，反感来自刻板男子气概。",
        "against_reason": "认为个人形象管理是自由，护肤与淡妆不应性别化，反感本身是对外貌与气质多样性的排斥。",
    },
    "反感苹果手机": {
        "description": "对苹果公司iPhone产品或品牌表达反感的消费舆论，涉及定价、封闭生态、创新节奏、维修成本与国货替代讨论。",
        "favor_reason": "认为定价过高、系统封闭、挤牙膏式升级或生态绑定体验差，反感亦含对品牌崇拜的逆反。",
        "against_reason": "认可其系统流畅、生态整合与长期口碑，反感多带情绪或民族主义消费，应客观比较产品本身。",
    },
}


def is_templated(entry: dict) -> bool:
    text = " ".join(
        entry.get(k, "") for k in ("description", "favor_reason", "against_reason")
    )
    return any(marker in text for marker in TEMPLATE_MARKERS)


def main() -> None:
    with TEMPLATE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    patched = 0
    skipped: list[str] = []

    for target, update in UPDATES.items():
        if target not in data:
            skipped.append(f"missing: {target}")
            continue
        entry = data[target]
        if not is_templated(entry):
            skipped.append(f"not templated: {target}")
            continue
        entry["description"] = update["description"]
        entry["favor_reason"] = update["favor_reason"]
        entry["against_reason"] = update["against_reason"]
        entry.pop("neutral_hint", None)
        patched += 1

    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    remaining = sum(1 for v in data.values() if is_templated(v))
    print(f"Patched: {patched}")
    print(f"UPDATES dict size: {len(UPDATES)}")
    print(f"Remaining templated (dislike markers): {remaining}")
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
