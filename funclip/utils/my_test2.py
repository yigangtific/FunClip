import re

def combine_subtitle_blocks_auto(text: str) -> str:
    """
    Automatically detect subtitle blocks that start with a line number
    and combine each block into a single line.
    Blocks are separated by blank lines in the output.
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    blocks = []
    current_block = []

    for line in lines:
        if not line:
            continue  # skip blank lines

        # Detect the start of a new block: line begins with digits + space
        if re.match(r'^\d+\s', line):
            # Save current block if it exists
            if current_block:
                blocks.append(" ".join(current_block))
                current_block = []
        current_block.append(line)

    # Add the last block
    if current_block:
        blocks.append(" ".join(current_block))

    # Join blocks with blank lines between them
    return "\n\n".join(blocks)


# ✅ Example usage
original_text = """
1
00:00:00,050 --> 00:00:01,450
In our inclusive design efforts

2
00:00:01,450 --> 00:00:05,850
There's a project I often talk about called "Finding the Beauty of Distant Places"

3
00:00:06,050 --> 00:00:06,395
Hello!
"""

revised_text = combine_subtitle_blocks_auto(original_text)
print(revised_text)
