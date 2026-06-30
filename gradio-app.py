import gradio as gr

from src.web_ui.layouts import InputBlock, OutputBlock
from src.web_ui.core import (
    checkhealth,
    fetch_available_models,
    load_denoising_ckpt,
    clean_speech,
    transcribe,
    plot_spectrogram,
)


def transcribe_and_plot(file, models):
    return transcribe(file, models), plot_spectrogram(file)

def perform_denoise(file, chosen_ckpt):
    global cur_ckpt
    if chosen_ckpt != cur_ckpt and load_denoising_ckpt(chosen_ckpt):
        cur_ckpt = chosen_ckpt
    return clean_speech(file)

def update_model_state(good_health):
    global cur_ckpt

    if good_health:
        models = fetch_available_models()
        cur_ckpt = models['loaded_denoiser']
        denoise_kwargs = {"choices": models['denoising_models'], "value": cur_ckpt}
        stt_kwargs = {"choices": models['stt_models'], "value": 'ChunkFormer'}
    else:
        denoise_kwargs, stt_kwargs = {}, {}

    return gr.update(**denoise_kwargs), gr.update(**stt_kwargs)

def render_page(good_health):
    return (gr.update(visible=good_health), gr.update(visible=not good_health))


cur_ckpt = None
with gr.Blocks(title="Speech Denoising Demonstration") as app:
    good_health = gr.State(False)
    original_trans_state = gr.State(value={})
    denoised_trans_state = gr.State(value={})

    with gr.Column(visible=False) as main_page:
        gr.HTML("<h1 id='main-title'>Speech Denoising Interface</h1>")

        with gr.Row():
            input_block = InputBlock()

        with gr.Column():
            output_block = OutputBlock(
                original_trans_state,
                denoised_trans_state
            )

    with gr.Column(visible=False) as down_page:
        with open("src/web_ui/down_page.html") as file:
            gr.HTML(file.read())

    app.load(fn=checkhealth, outputs=good_health).then(
        fn=render_page,
        inputs=good_health,
        outputs=[main_page, down_page]
    ).then(
        fn=update_model_state,
        inputs=good_health,
        outputs=[input_block.model_dropdown, input_block.stt_model_dropdown]
    )

    input_block.submit_btn.click(
        fn=perform_denoise,
        inputs=[input_block.audio_input, input_block.model_dropdown],
        outputs=[output_block.audio_output]
    ).then(
        fn=transcribe_and_plot,
        inputs=[output_block.audio_output, input_block.stt_model_dropdown],
        outputs=[denoised_trans_state, output_block.denoised_spec]
    )

    input_block.submit_btn.click(
        fn=transcribe_and_plot,
        inputs=[input_block.audio_input, input_block.stt_model_dropdown],
        outputs=[original_trans_state, output_block.original_spec]
    )

app.launch(
    css_paths=["src/web_ui/style.css"],
    debug=True
)
