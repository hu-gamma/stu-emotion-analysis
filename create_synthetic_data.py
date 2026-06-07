"""
创建合成数据集用于快速训练演示模型
"""
import os
import random
import pandas as pd

# 每个情感类别的模板句子
TEMPLATES = {
    '悲伤': [
        "今天心情很糟糕，感觉一切都不顺利。",
        "听到这个消息我真的很伤心。",
        "最近总是感到很失落，不知道该怎么办。",
        "考试没考好，心里很难受。",
        "和朋友吵架了，现在很沮丧。",
        "看到那个场景我忍不住哭了。",
        "生活好难，我感觉坚持不下去了。",
        "失去了重要的东西，心里空落落的。",
        "今天被批评了，感到很委屈。",
        "好想念以前的日子，现在一切都变了。",
        "努力了很久却没有结果，好失望。",
        "独自一人的时候总是感到很难过。",
        "为什么事情总是朝坏的方向发展。",
        "心里像压了一块大石头，喘不过气来。",
        "看到伤感的新闻，心情变得很低落。",
    ],
    '快乐': [
        "今天真是太开心了！",
        "终于完成了这个项目，感觉棒极了！",
        "和朋友出去玩了一整天，超级开心。",
        "收到录取通知书的那一刻我激动得跳了起来。",
        "今天的晚餐太好吃了，幸福感爆棚。",
        "老师表扬了我，心里美滋滋的。",
        "周末要去旅行，期待了好久。",
        "和喜欢的人聊天真的很开心。",
        "今天的天气真好，心情也跟着好起来了。",
        "买到了心仪已久的东西，开心！",
        "团队协作很顺利，大家都很有成就感。",
        "生日收到了很多祝福，感到被爱包围。",
        "运动完出了一身汗，感觉特别爽。",
        "听到喜欢的歌，忍不住跟着哼起来。",
        "成功解决了一个难题，太有成就感了。",
    ],
    '愤怒': [
        "这简直太不公平了，我很生气！",
        "为什么总是这样，我真的受够了！",
        "这种行为真的很让人火大。",
        "被欺骗的感觉真的很愤怒。",
        "排队排了一个小时，结果被告知没有了，气死我了。",
        "别人抄袭我的作品还理直气壮，太可恶了。",
        "说话不算数的人最让人讨厌。",
        "被误解的感觉真的很憋屈。",
        "明明不是我的错，却要我来承担后果。",
        "这种不负责任的态度让我很恼火。",
        "一次次被放鸽子，真的很生气。",
        "看到 bully 的行为，我感到非常愤怒。",
        "努力被否定，真的很不甘心。",
        "为什么有些人总是这么自私。",
        "被不公平对待的感觉真的很差。",
    ],
    '焦虑': [
        "明天就要考试了，我现在很紧张。",
        "不知道结果会怎样，心里七上八下的。",
        "截止日期快到了，我还有好多没做完。",
        "面试前总是感到忐忑不安。",
        "等待结果的过程真的很煎熬。",
        "担心自己做不好，压力好大。",
        "最近总是睡不好，脑子里想太多事情。",
        "不确定未来会怎样，感到很迷茫。",
        "害怕让别人失望，所以总是很紧张。",
        "时间不够用，急得手心都出汗了。",
        "第一次做这件事，心里没底。",
        "邮件发出去之后一直在担心对方的回复。",
        "总觉得自己准备得不够充分。",
        "好多事情堆在一起，感到不知所措。",
        " deadline 逼近的感觉让人喘不过气。",
    ],
    '厌恶': [
        "这种行为真的很恶心。",
        "我完全无法接受这样的事情。",
        "看到那个场面我感到很不舒服。",
        "这种人真的让人反感。",
        "这种味道我实在受不了。",
        "虚伪的行为让我感到厌恶。",
        "看到不文明的行为就觉得很反感。",
        "那种声音听得我起鸡皮疙瘩。",
        "这种不诚实的做法让人很失望。",
        "我实在看不惯这种double standard。",
        "卫生状况太差了，感到很不舒服。",
        "这种功利主义的价值观让人反感。",
        "背后说人坏话的行为真让人厌恶。",
        "抄袭别人的成果还装作是自己的，恶心。",
        "那种高高在上的态度让人很反感。",
    ],
    '惊讶': [
        "哇，真的吗？太意外了！",
        "这完全出乎我的意料！",
        "竟然会发生这种事，太神奇了。",
        "听到这个消息我惊呆了。",
        "没想到结果会是这样，好惊讶。",
        "这也太巧了吧，不可思议。",
        "突然得知这个消息，一时反应不过来。",
        "这个转折让我大吃一惊。",
        "原来真相是这样的，太震惊了。",
        "我从来没有想过会是这种情况。",
        "这个发现让我觉得很意外。",
        "他竟然做出了这样的决定，令人惊讶。",
        "出乎意料的好结果，太惊喜了。",
        "这种巧合简直难以置信。",
        "看到结果的时候我的眼睛都瞪大了。",
    ],
    '好奇': [
        "这是为什么呢？我想知道答案。",
        "这个现象好有趣，我想深入研究一下。",
        "能不能告诉我更多细节？",
        "我对这个领域很感兴趣，想多了解一下。",
        "这个问题困扰了我很久，一直想不明白。",
        "看到新奇的东西就忍不住想探索。",
        "这个原理是什么？好想知道。",
        "世界上怎么会有这么奇妙的事情。",
        "我想知道背后的故事。",
        "这本书激起了我强烈的求知欲。",
        "这个实验结果很有意思，想验证一下。",
        "对未知的事物总是充满好奇。",
        "这个设计思路是怎么想出来的？",
        "想了解更多关于这个话题的内容。",
        "第一次听说这个概念，感觉很有意思。",
    ],
}


def generate_single_turn_dataset(output_path, n_per_class=200):
    """生成单轮对话数据集"""
    rows = []
    for emotion, templates in TEMPLATES.items():
        for i in range(n_per_class):
            # 从模板中随机选择，并做简单变体
            template = random.choice(templates)
            text = template
            rows.append({
                'transformed_text': text,
                'fine_grained_emotion': emotion,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"单轮数据集已生成: {output_path}, 共 {len(df)} 条")
    return df


def generate_multiturn_dataset(output_path, n_samples=500):
    """生成多轮对话数据集"""
    rows = []

    # 生成对话上下文对
    for _ in range(n_samples):
        # 随机选择两个情感
        emotion1 = random.choice(list(TEMPLATES.keys()))
        emotion2 = random.choice(list(TEMPLATES.keys()))

        text1 = random.choice(TEMPLATES[emotion1])
        text2 = random.choice(TEMPLATES[emotion2])

        # 有50%概率有上下文
        if random.random() < 0.5:
            prev = text1
            curr = text2
            emotion = emotion2
        else:
            prev = ""
            curr = text2
            emotion = emotion2

        rows.append({
            'prev_text': prev,
            'curr_text': curr,
            'emotion': emotion,
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"多轮数据集已生成: {output_path}, 共 {len(df)} 条")
    return df


if __name__ == '__main__':
    os.makedirs('/mnt/stu-emotion-analysis/datasets/csv', exist_ok=True)

    # 生成单轮数据集
    df_single = generate_single_turn_dataset(
        '/mnt/stu-emotion-analysis/datasets/csv/synthetic_single_turn.csv',
        n_per_class=200
    )

    # 生成多轮数据集
    df_multi = generate_multiturn_dataset(
        '/mnt/stu-emotion-analysis/datasets/csv/synthetic_multiturn.csv',
        n_samples=800
    )

    print("\n数据集统计:")
    print("\n单轮数据集:")
    print(df_single['fine_grained_emotion'].value_counts())
    print("\n多轮数据集:")
    print(df_multi['emotion'].value_counts())
