import gradio as gr


class InputBlock:
    def __init__(
        self,
        denoising_model_choices=None,
        stt_model_choices=None,
    ) -> None:
        with gr.Column(scale=3):
            self.audio_input = gr.Audio(type="filepath", label="Upload Audio")

        with gr.Column(scale=1):
            self.model_dropdown = gr.Dropdown(
                choices=denoising_model_choices, 
                label="Checkpoint",
                interactive=True
            )
            self.stt_model_dropdown = gr.Dropdown(
                choices=stt_model_choices, 
                label="Speech-to-text Models",
                multiselect=True,
                interactive=True
            )
            self.submit_btn = gr.Button("Process Audio", variant="primary")


class OutputBlock:
    def __init__(
        self,
        original_transcripts_state: gr.State,
        denoised_transcripts_state: gr.State
    ) -> None:
        self._title = gr.Markdown("## Denoising Output")
        with gr.Column():
            self.audio_output = gr.Audio(
                label="Processed Output",
                type='filepath',
                interactive=False
            )

            with gr.Column():
                gr.render(inputs=[
                    original_transcripts_state,
                    denoised_transcripts_state,
                ])(self.render_transcription_tabs)

            with gr.Row():
                self.original_spec = gr.Plot(label="Original (dB)")
                self.denoised_spec = gr.Plot(label="Denoised (dB)")

    def _create_transcription_row(
        self,
        orig_placeholder=None,
        denoised_placeholder=None,
        orig_value=None,
        denoised_value=None,
        orig_label="Original Transcription",
        denoised_label="Denoised Transcription",
    ):
        with gr.Row():
            gr.Textbox(
                orig_value,
                label=orig_label,
                placeholder=orig_placeholder,
                interactive=False,
                lines=4,
                elem_id="transcript-box"
            )
            gr.Textbox(
                denoised_value,
                label=denoised_label,
                placeholder=denoised_placeholder,
                interactive=False,
                lines=4,
                elem_id="transcript-box"
            )

    def render_transcription_tabs(self, orig_dict, denoised_dict):
        if not orig_dict and not denoised_dict:
            self._create_transcription_row(
                "Transcription of original speech...",
                "Transcription of processed speech...",
            )
            return

        all_models = list(set(orig_dict.keys()) | set(denoised_dict.keys()))
        with gr.Tabs():
            for model in all_models:
                with gr.Tab(label=model):
                    self._create_transcription_row(
                        orig_value=orig_dict.get(model, ""),
                        denoised_value=denoised_dict.get(model, "")
                    )
