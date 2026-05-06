GPUS=(0 1 2 3 4 5 6 7)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_dir="${MIMIC_VIDEO_CHECKPOINT_DIR:-$(cd "${SCRIPT_DIR}/../../model" && pwd)/checkpoints}"

fifo="/tmp/gpuq.$$"
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"

for i in $(seq 0 $(( 2 * ${#GPUS[@]} - 1 ))); do
  g=$((i % ${#GPUS[@]}))
  echo "$g" >&3
done

launch() {
  local gpu
  read -u 3 gpu || exit 1
  (
    if [[ "$gpu" -eq 0 ]]; then
      export CUDA_VISIBLE_DEVICES=0
      export SAPIEN_RENDER_CUDA_ORDINAL=0
    else
      export CUDA_VISIBLE_DEVICES="${gpu},0"
      export SAPIEN_RENDER_CUDA_ORDINAL=1
    fi
    export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select

    "$@"
    rc=$?
    echo "$gpu" >&3
    exit $rc
  ) &
}

declare -A ptcosmos=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256
  [action_model]=${checkpoint_dir}/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_pretrained_cosmos.pt
  [stats]=${checkpoint_dir}/dataset_statistics/bridge.json
)

declare -A ftcosmos=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256
  [action_model]=${checkpoint_dir}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/bridge.json
)

models=(ptcosmos ftcosmos)

execute_actions=(5)
stop_steps=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)

for model in "${models[@]}"; do
  declare -n M=$model

  action_stem="$(basename -- "${M[action_model]}")"
  action_stem="${action_stem%.pt}"

  for execute_steps in "${execute_actions[@]}"; do
    for stop in "${stop_steps[@]}"; do
    variation_name="${action_stem}_stopafter${stop}_execute${execute_steps}"

      scene_name=bridge_table_1_v1
      robot=widowx
      rgb_overlay_path=SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
      robot_init_x=0.147
      robot_init_y=0.028

      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 8 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 8 16 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 16 24 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";

      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 8 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 8 16 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 16 24 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";

      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 8 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 8 16 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
        --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 16 24 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";

      scene_name=bridge_table_1_v2
      robot=widowx_sink_camera_setup
      rgb_overlay_path=SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
      robot_init_x=0.127
      robot_init_y=0.06

      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
        --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 8 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
        --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 8 16 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";
      launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path ${variation_name} \
        --robot ${robot} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
        --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 16 24 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --vam-experiment-name "${M[experiment_name]}" \
        --vam-video-model-path "${M[video_model]}" \
        --vam-action-model-path "${M[action_model]}" \
        --vam-dataset-statistics-path "${M[stats]}" \
        --vam-num-execute-actions "${execute_steps}" \
        --vam-img-horizon "${M[img_h]}" \
        --vam-lowdim-horizon "${M[lowdim_h]}" \
        --vam-stop-video-denoising-step "${stop}";

    done
  done
done

wait
exec 3>&- 3<&-
