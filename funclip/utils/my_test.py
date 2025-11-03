def combine_subtitle_blocks(text: str) -> str:
    # Split text into lines and remove leading/trailing spaces
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    combined = []
    # Process in chunks of 3 lines
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            number = lines[i]
            timestamp = lines[i + 1]
            content = lines[i + 2]
            combined.append(f"{number} {timestamp} {content}")

    # Join each paragraph with a blank line
    return "\n\n".join(combined)


# Example input
original_text = """
1
00:00:00,050 --> 00:00:01,450
在我们的设计普惠当中

2
00:00:01,450 --> 00:00:05,850
有一个我经常津津乐道的项目叫寻找远方的美好

3
00:00:06,050 --> 00:00:06,395
啊

4
00:00:07,120 --> 00:00:10,480
在这样一个我们叫寻美在这样的一个项目当中

5
00:00:10,660 --> 00:00:12,940
我们把它跟乡村振兴去结合起来

6
00:00:13,240 --> 00:00:15,140
利用我们的设计的能力

7
00:00:15,260 --> 00:00:17,480
问我们自身员工的设设计能力

8
00:00:17,560 --> 00:00:19,380
我们设计生态伙伴的能力

9
00:00:19,600 --> 00:00:21,840
帮助乡村振兴当中

10
00:00:22,620 --> 00:00:25,620
要希望把他的产品推向市场

11
00:00:25,620 --> 00:00:30,880
把他的农产品把他加工产品推向市场的这样的伙伴做一件事情

12
00:00:31,120 --> 00:00:33,575
就是帮他们设计一个好的包装物
"""

# Run the function
revised_text = combine_subtitle_blocks(original_text)

print(revised_text)
