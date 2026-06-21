import gradio as gr

from src.web_ui.core import clean_speech, transcribe, plot_spectrogram


def transcribe_and_plot(file):
    return transcribe(file), plot_spectrogram(file)


with gr.Blocks() as app:
    gr.HTML("<h1 id='main-title'>Speech Denoising Interface</h1>")

    with gr.Column():

        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            submit_btn = gr.Button("Process Audio", variant="primary")

        gr.Markdown("## Denoising Output")
        with gr.Column():
            audio_output = gr.Audio(
                label="Processed Output",
                type='filepath'
            )

            with gr.Row():
                original_transcript = gr.Textbox(
                    label="Original Transcription",
                    interactive=False,
                    lines=5,
                    placeholder="Transcription of original speech...",
                    elem_id="transcript-box"
                )
                denoised_transcript = gr.Textbox(
                    label="Denoised Transcription",
                    interactive=False,
                    lines=5,
                    placeholder="Transcription of processed speech...",
                    elem_id="transcript-box"
                )

            with gr.Row():
                original_spec = gr.Plot(label="Original (dB)")
                denoised_spec = gr.Plot(label="Denoised (dB)")

    submit_btn.click(
        fn=clean_speech,
        inputs=audio_input,
        outputs=[audio_output]
    ).then(
        fn=transcribe_and_plot,
        inputs=audio_output,
        outputs=[denoised_transcript, denoised_spec]
    )

    submit_btn.click(
        fn=transcribe_and_plot,
        inputs=audio_input,
        outputs=[original_transcript, original_spec]
    )

app.launch(
    css_paths=["src/web_ui/style.css"],
    debug=True
)
