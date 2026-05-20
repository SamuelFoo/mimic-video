import torch
from torch import nn

from cosmos_predict2.pipelines.video2world import DeviceMesh, Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline


class Video2World2ActionPipeline(nn.Module):
    def __init__(
        self,
        video2world_pipeline: Video2WorldPipeline,
        world2action_pipeline: World2ActionPipeline,
    ) -> None:
        super().__init__()

        self.video2world_pipeline = video2world_pipeline
        self.world2action_pipeline = world2action_pipeline

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.video2world_pipeline.apply_fsdp(dp_mesh)
        self.world2action_pipeline.apply_fsdp(dp_mesh)

    def apply_cp(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(
        self,
        input_vid: torch.Tensor,
        state_B_HO_O: torch.Tensor,
        prompt: str,
        prompt_embedding: torch.Tensor | None = None,
        num_sampling_step: int = 35,
        stop_after_step: int | None = None,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor:
        if self.video2world_pipeline.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            if not guardrail_presets.run_text_guardrail(prompt, self.video2world_pipeline.text_guardrail_runner):
                msg = "Text guardrail error on prompt."
                raise RuntimeError(msg)

        T = input_vid.shape[2]
        assert T in {1, 5}

        crossattn_emb, video_sigma = self.video2world_pipeline.generate_video(
            vid_input=input_vid,
            num_latent_conditional_frames=1 if T == 1 else 2,
            prompt=prompt,
            prompt_embedding=prompt_embedding,
            negative_prompt="",
            guidance=0.0,
            num_sampling_step=num_sampling_step,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            return_context_at_step=stop_after_step,
            hidden_state_layer_idx=self.world2action_pipeline.config.xattn_layer_idx,
        )

        # Match training: slice the video DiT's hidden_states to the first N
        # latent timesteps before feeding into the action decoder. Must equal
        # the value used during training.
        if self.world2action_pipeline.config.xattn_video_prefix_length is not None:
            crossattn_emb = crossattn_emb[:, : self.world2action_pipeline.config.xattn_video_prefix_length]

        hidden_state_shape = crossattn_emb.shape
        crossattn_emb = crossattn_emb.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])

        return self.world2action_pipeline(
            state_B_HO_O=state_B_HO_O,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma.unsqueeze(1),
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
