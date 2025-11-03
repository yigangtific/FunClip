import gradio as gr

def generate_checkboxes(text):
    """
    Dynamically generate a list of checkbox labels based on TextBox content.
    Returns a list of gr.Checkbox updates for the container.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []  # empty list will clear the container
    # Return list of gr.Checkbox objects
    return [gr.Checkbox.update(label=line, value=False, visible=True) for line in lines]

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Dynamic Checkboxes per Line")

        text_input = gr.Textbox(
            label="Enter lines for checkboxes (one line per checkbox)",
            placeholder="Apple\nBanana\nCherry",
            lines=5
        )

        # Container for dynamic checkboxes
        checkbox_container = gr.Column()

        # Create some placeholders initially
        placeholder_checkboxes = []
        for i in range(10):  # maximum 10 checkboxes for placeholder
            cb = gr.Checkbox(label="", visible=False)
            placeholder_checkboxes.append(cb)

        # When TextBox changes, update each placeholder checkbox
        def update_checkboxes(text):
            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            updates = []
            for i in range(len(placeholder_checkboxes)):
                if i < len(lines):
                    updates.append(gr.Checkbox.update(label=lines[i], value=False, visible=True))
                else:
                    updates.append(gr.Checkbox.update(label="", value=False, visible=False))
            return updates

        text_input.change(
            fn=update_checkboxes,
            inputs=text_input,
            outputs=placeholder_checkboxes
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
