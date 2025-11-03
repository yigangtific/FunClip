import gradio as gr
import re

# Example subtitle input
subtitle_text = """1 00:00:00,050 --> 00:00:01,450 On our Earth
2 00:00:01,450 --> 00:00:05,850 There is a country located in Africa
3 00:00:06,050 --> 00:00:06,395 Hello"""

# Parse each line into (number, timestamp, text)
def parse_subtitles(text):
    lines = text.strip().splitlines()
    parsed = []
    for line in lines:
        match = re.match(r'^(\d+)\s+([\d:,>\s-]+)\s+(.+)$', line.strip())
        if match:
            num = int(match.group(1))
            timestamp = match.group(2).strip()
            content = match.group(3).strip()
            parsed.append((num, timestamp, content))
    return parsed


# Combine text for selected checkboxes.
# We accept variable positional args; the last arg is expected to be `lines` (the gr.State),
# while the preceding args are checkbox boolean values.
def combine_selected(*args):
    if not args:
        return ""
    # Heuristic: last arg is parsed lines if it's a list of tuples
    last = args[-1]
    if isinstance(last, list) and last and isinstance(last[0], tuple):
        lines = last
        checkbox_values = args[:-1]
    else:
        # If last arg isn't lines, assume all args are checkbox values and we can't proceed
        checkbox_values = args
        lines = None

    if lines is None:
        return ""

    # Build list of selected indices (1-based numbers from parsed lines)
    selected_nums = [num for (num, val) in zip(range(1, len(lines) + 1), checkbox_values) if val]
    if not selected_nums:
        return ""

    selected_nums.sort()
    num_to_text = {num: content for num, _, content in lines}
    combined = ". ".join([num_to_text[num] for num in selected_nums])
    return combined.strip()


# Button handlers
def select_all(lines):
    """Set all checkboxes to True (returned list length must match number of checkboxes)."""
    return [True] * len(lines)

def deselect_all(lines):
    """Set all checkboxes to False."""
    return [False] * len(lines)


# Build Gradio interface
def build_interface():
    parsed_lines = parse_subtitles(subtitle_text)

    with gr.Blocks() as demo:
        gr.Markdown("### 🎬 Subtitle Line Selector")
        gr.Markdown("Each subtitle line has its own checkbox. Use the buttons below to select or deselect all lines.")

        # Action buttons
        with gr.Row():
            select_all_btn = gr.Button("✅ Select All")
            deselect_all_btn = gr.Button("❌ Deselect All")

        # Create checkbox components dynamically (each in its own row)
        checkboxes = []
        with gr.Column():
            for num, timestamp, content in parsed_lines:
                label_text = f"{num}. {timestamp} {content}"
                cb = gr.Checkbox(label=label_text, value=False)
                checkboxes.append(cb)

        output = gr.Textbox(label="Merged Result", interactive=True, lines=3)

        # Update merged result when any checkbox changes
        # Inputs: all checkbox components followed by gr.State(parsed_lines)
        for cb in checkboxes:
            cb.change(
                fn=combine_selected,
                inputs=checkboxes + [gr.State(parsed_lines)],
                outputs=output
            )

        # Button click events: first update checkboxes, then recompute output.
        select_all_btn.click(
            fn=select_all,
            inputs=[gr.State(parsed_lines)],
            outputs=checkboxes
        )
        deselect_all_btn.click(
            fn=deselect_all,
            inputs=[gr.State(parsed_lines)],
            outputs=checkboxes
        )

        # After buttons change the checkboxes, also recompute the merged output.
        # We wire the same buttons to call combine_selected with the full checkbox list + state.
        # Because Gradio runs both click handlers, the order is preserved: checkboxes will
        # be updated first (because their outputs are checkboxes), and then this second handler
        # computes using the (new) checkbox values provided by the client.
        select_all_btn.click(
            fn=combine_selected,
            inputs=checkboxes + [gr.State(parsed_lines)],
            outputs=output
        )
        deselect_all_btn.click(
            fn=combine_selected,
            inputs=checkboxes + [gr.State(parsed_lines)],
            outputs=output
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
