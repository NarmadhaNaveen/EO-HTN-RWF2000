
import os
import sys
import time
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch


# ==================================================
# 1. PROJECT PATH AND IMPORTS
# ==================================================

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.eo_htn import EOHTN
from utils.temporal_analysis import analyze_segments


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EO-HTN | Intelligent CCTV Analysis",
    page_icon="🎥",
    layout="wide"
)

torch.set_num_threads(2)

CHECKPOINT_PATH = os.path.join(
    ROOT,
    "experiments",
    "eo_htn_v2",
    "checkpoints",
    "best_eo_htn_v2.pth"
)

V2_LOG_DIR = os.path.join(
    ROOT,
    "experiments",
    "eo_htn_v2",
    "logs"
)


# ==================================================
# 3. ABSTRACT TARGETS AND ACTUAL V2 RESULTS
# ==================================================

ABSTRACT_TARGETS = {
    "Accuracy (%)": 93.4,
    "F1-score (%)": 93.4,
    "Parameters (M)": 7.4,
    "Brier Score": 0.078,
    "ECE (%)": 3.2
}

ACTUAL_V2_RESULTS = {
    "Accuracy (%)": 71.25,
    "F1-score (%)": 77.31755424,
    "Parameters (M)": 0.411426,
    "Brier Score": 0.378133,
    "ECE (%)": 6.0061
}

V2_CONFUSION_MATRIX = np.array([
    [89, 111],
    [4, 196]
])

V2_BEST_EPOCH = 9
V2_TOTAL_EPOCHS = 14
V2_VALIDATION_VIDEOS = 400
V2_TRAINING_VIDEOS = 1600

V2_THRESHOLD = 0.45

V2_THRESHOLD_RESULTS = {
    "Accuracy": 71.25,
    "Precision": 63.67,
    "Recall": 99.00,
    "F1-score": 77.50,
    "Specificity": 43.50,
    "False Positives": 113,
    "False Negatives": 2
}


# ==================================================
# 4. MOUNTAIN-STYLE PLOTLY CHART FUNCTION
# ==================================================

def mountain_chart(
    x,
    y,
    title,
    color,
    y_title,
    chart_key
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=title,
            line=dict(
                color=color,
                width=3,
                shape="spline",
                smoothing=1.1
            ),
            fill="tozeroy",
            fillcolor=color.replace(
                "1)",
                "0.20)"
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title=y_title,
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=40
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key
    )


# ==================================================
# 5. LOAD TRAINED V2 MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            "V2 checkpoint not found: " + CHECKPOINT_PATH
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    frames = checkpoint["frames"]
    size = checkpoint["size"]

    model = EOHTN(
        frames=frames
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        model,
        frames,
        size,
        parameter_count
    )


# ==================================================
# 6. VIDEO FRAME EXTRACTION
# ==================================================

def extract_video_frames(
    video_path,
    number_of_frames,
    size
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open uploaded video."
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()

        raise RuntimeError(
            "Video has no readable frames."
        )

    positions = np.linspace(
        0,
        total_frames - 1,
        number_of_frames
    ).astype(int)

    frames = []
    processed_images = []
    failed_frames = 0

    for position in positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(position)
        )

        success, frame = cap.read()

        if not success:

            failed_frames += 1

            rgb = np.zeros(
                (size, size, 3),
                dtype=np.uint8
            )

        else:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

        display_frame = cv2.resize(
            rgb,
            (320, 240)
        )

        model_frame = cv2.resize(
            rgb,
            (size, size)
        )

        frames.append(
            display_frame
        )

        processed_images.append(
            model_frame
        )

    cap.release()

    array = np.stack(
        processed_images
    ).astype(np.float32) / 255.0

    array = (array - 0.5) / 0.5

    tensor = torch.from_numpy(
        array
    ).permute(
        0, 3, 1, 2
    ).contiguous().unsqueeze(0)

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    timestamps = (
        positions / fps
        if fps > 0
        else positions.astype(float)
    )

    return (
        tensor,
        frames,
        timestamps,
        total_frames,
        fps,
        duration,
        failed_frames
    )


# ==================================================
# 7. EO-HTN INFERENCE AND CONTEXT GATE ANALYSIS
# ==================================================

def analyze_video(
    model,
    video_tensor
):

    captured = {}

    def capture_gate_input(
        module,
        inputs
    ):

        captured["features"] = (
            inputs[0].detach()
        )

    hook = model.context_gate.register_forward_pre_hook(
        capture_gate_input
    )

    start_time = time.perf_counter()

    try:

        with torch.inference_mode():

            logits = model(
                video_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            if "features" not in captured:

                raise RuntimeError(
                    "Could not capture context-gate features."
                )

            features = captured["features"]

            context = features.mean(
                dim=1,
                keepdim=True
            ).expand_as(features)

            gate_weights = model.context_gate.gate(
                torch.cat(
                    [features, context],
                    dim=-1
                )
            )

            importance = gate_weights.mean(
                dim=-1
            )[0]

    finally:

        hook.remove()

    inference_time = (
        time.perf_counter() - start_time
    )

    return (
        probabilities.cpu().numpy(),
        importance.cpu().numpy(),
        inference_time
    )


# ==================================================
# 8. DASHBOARD STYLING
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .sub-title {
        color: #8b9bb4;
        font-size: 15px;
        letter-spacing: 2px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🎥 EO-HTN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'EDGE-OPTIMIZED HYBRID TRANSFORMER NETWORK'
    '</div>',
    unsafe_allow_html=True
)

st.subheader(
    "Context-Aware Threat Classification "
    "in Visual Security Systems"
)

st.caption(
    "Research Prototype | RWF-2000 | Fight / NonFight Classification"
)

st.divider()


# ==================================================
# 9. LOAD MODEL
# ==================================================

try:

    (
        model,
        MODEL_FRAMES,
        MODEL_SIZE,
        PARAMETER_COUNT
    ) = load_model()

    st.success(
        "EO-HTN V2 trained checkpoint loaded successfully."
    )

except Exception as error:

    st.error(
        "Model loading failed: " + str(error)
    )

    st.stop()


# ==================================================
# 10. RESEARCH PERFORMANCE DASHBOARD
# ==================================================

st.header(
    "1. Research Performance Dashboard"
)

st.caption(
    "Proposed targets from the research abstract compared "
    "with measured results from the V2 validation evaluation."
)

st.info(
    "Abstract values are research targets. Actual values "
    "are measured V2 validation results. These must not be "
    "interpreted as achieved research targets."
)

st.subheader(
    "Actual V2 Performance"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    "71.25%",
    "Target: 93.40%",
    delta_color="off"
)

c2.metric(
    "F1-score",
    "77.32%",
    "Target: 93.40%",
    delta_color="off"
)

c3.metric(
    "Trainable Parameters",
    f"{PARAMETER_COUNT / 1_000_000:.3f}M",
    "Target: 7.4M",
    delta_color="off"
)

c4, c5 = st.columns(2)

c4.metric(
    "Brier Score",
    "0.3781",
    "Target: 0.078",
    delta_color="off"
)

c5.metric(
    "Expected Calibration Error",
    "6.01%",
    "Target: 3.20%",
    delta_color="off"
)

st.divider()


# ==================================================
# 11. PERFORMANCE COMPARISON
# ==================================================

st.subheader(
    "Abstract Target vs Actual V2 Results"
)

comparison_df = pd.DataFrame({
    "Metric": list(ABSTRACT_TARGETS.keys()),

    "Abstract Target": list(
        ABSTRACT_TARGETS.values()
    ),

    "Actual V2 Result": [
        ACTUAL_V2_RESULTS[key]
        for key in ABSTRACT_TARGETS.keys()
    ]
})

comparison_df["Difference"] = (
    comparison_df["Actual V2 Result"]
    - comparison_df["Abstract Target"]
)

st.dataframe(
    comparison_df.round(4),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "For Brier Score and ECE, lower values are better. "
    "The parameter count is a model-size comparison, "
    "not an accuracy metric."
)

st.divider()


# ==================================================
# 12. CLASSIFICATION PERFORMANCE COMPARISON
# ==================================================

st.subheader(
    "Classification Performance Comparison"
)

classification_comparison = pd.DataFrame(
    {
        "Abstract Target": [
            93.4,
            93.4
        ],

        "Actual V2 Result": [
            71.25,
            77.31755424
        ]
    },
    index=[
        "Accuracy (%)",
        "F1-score (%)"
    ]
)

st.bar_chart(
    classification_comparison,
    use_container_width=True
)

st.divider()


# ==================================================
# 13. CALIBRATION COMPARISON
# ==================================================

st.subheader(
    "Model Calibration Comparison"
)

calibration_comparison = pd.DataFrame(
    {
        "Abstract Target": [
            0.078,
            3.2
        ],

        "Actual V2 Result": [
            0.378133,
            6.0061
        ]
    },
    index=[
        "Brier Score",
        "ECE (%)"
    ]
)

st.bar_chart(
    calibration_comparison,
    use_container_width=True
)

st.caption(
    "Lower Brier Score and ECE values indicate better "
    "probability calibration."
)

st.divider()


# ==================================================
# 14. VALIDATION CONFUSION MATRIX
# ==================================================

st.subheader(
    "V2 Validation Confusion Matrix"
)

st.caption(
    "Rows represent actual classes. Columns represent "
    "predicted classes. Class order: NonFight, Fight."
)

confusion_df = pd.DataFrame(
    V2_CONFUSION_MATRIX,
    index=[
        "Actual NonFight",
        "Actual Fight"
    ],
    columns=[
        "Predicted NonFight",
        "Predicted Fight"
    ]
)

st.dataframe(
    confusion_df,
    use_container_width=True
)

tn, fp = V2_CONFUSION_MATRIX[0]
fn, tp = V2_CONFUSION_MATRIX[1]

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "True Negatives",
    int(tn)
)

m2.metric(
    "False Positives",
    int(fp)
)

m3.metric(
    "False Negatives",
    int(fn)
)

m4.metric(
    "True Positives",
    int(tp)
)

st.warning(
    "The V2 validation model has 111 false positives. "
    "Some NonFight videos were incorrectly classified "
    "as Fight. Consider this limitation when discussing "
    "model reliability."
)

st.divider()


# ==================================================
# 15. V2 TRAINING INFORMATION
# ==================================================

st.subheader(
    "V2 Experiment Information"
)

t1, t2, t3, t4 = st.columns(4)

t1.metric(
    "Training Videos",
    V2_TRAINING_VIDEOS
)

t2.metric(
    "Validation Videos",
    V2_VALIDATION_VIDEOS
)

t3.metric(
    "Best Epoch",
    V2_BEST_EPOCH
)

t4.metric(
    "Completed Epochs",
    V2_TOTAL_EPOCHS
)

t5, t6, t7 = st.columns(3)

t5.metric(
    "Input Frames",
    MODEL_FRAMES
)

t6.metric(
    "Input Resolution",
    f"{MODEL_SIZE} × {MODEL_SIZE}"
)

t7.metric(
    "Parameters",
    f"{PARAMETER_COUNT:,}"
)

st.caption(
    "The best V2 checkpoint was selected using validation "
    "F1-score. Training stopped early after 14 epochs."
)

st.divider()


# ==================================================
# 16. THRESHOLD ANALYSIS
# ==================================================

st.subheader(
    "V2 Threshold Analysis"
)

st.caption(
    "Validation threshold analysis using saved V2 validation "
    "predictions. These are exploratory validation results, "
    "not independent test results."
)

th1, th2, th3 = st.columns(3)

th1.metric(
    "Selected Grid Threshold",
    "0.45"
)

th2.metric(
    "Accuracy at 0.45",
    "71.25%"
)

th3.metric(
    "F1-score at 0.45",
    "77.50%"
)

threshold_df = pd.DataFrame({
    "Metric": [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1-score (%)",
        "Specificity (%)"
    ],

    "Result": [
        71.25,
        63.67,
        99.00,
        77.50,
        43.50
    ]
})

st.dataframe(
    threshold_df,
    use_container_width=True,
    hide_index=True
)

st.write(
    "**False Positives:** 113"
)

st.write(
    "**False Negatives:** 2"
)

st.warning(
    "Threshold 0.45 was selected by examining the same "
    "validation predictions. Do not treat it as a final "
    "deployment threshold without independent testing."
)

st.divider()


# ==================================================
# 17. VIDEO UPLOAD
# ==================================================

st.header(
    "2. Upload CCTV Video"
)

uploaded_video = st.file_uploader(
    "Select a video for threat classification",
    type=[
        "avi",
        "mp4",
        "mov"
    ]
)

if uploaded_video is None:

    st.info(
        "Upload a CCTV video to begin the analysis."
    )

    st.markdown(
        "### Current V2 system configuration"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        "EO-HTN V2"
    )

    c2.metric(
        "Input Frames",
        MODEL_FRAMES
    )

    c3.metric(
        "Image Size",
        f"{MODEL_SIZE} × {MODEL_SIZE}"
    )

    st.stop()


# ==================================================
# 18. VIDEO PREVIEW
# ==================================================

st.video(
    uploaded_video
)

st.write(
    "**Filename:**",
    uploaded_video.name
)

st.write(
    "**Upload size:**",
    f"{uploaded_video.size / (1024 * 1024):.2f} MB"
)

st.divider()


# ==================================================
# 19. ANALYZE BUTTON
# ==================================================

if st.button(
    "🔍 Analyze Video",
    type="primary",
    use_container_width=True
):

    temp_path = None

    try:

        with st.spinner(
            "Extracting video frames and running EO-HTN V2..."
        ):

            temp_dir = os.path.join(
                ROOT,
                "_temp"
            )

            os.makedirs(
                temp_dir,
                exist_ok=True
            )

            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(
                    uploaded_video.name
                )[1],
                dir=temp_dir,
                delete=False
            ) as temp_file:

                temp_file.write(
                    uploaded_video.getvalue()
                )

                temp_path = temp_file.name

            (
                video_tensor,
                display_frames,
                timestamps,
                total_frames,
                fps,
                duration,
                failed_frames
            ) = extract_video_frames(
                temp_path,
                MODEL_FRAMES,
                MODEL_SIZE
            )

            (
                probabilities,
                importance,
                inference_time
            ) = analyze_video(
                model,
                video_tensor
            )

            segment_results = analyze_segments(
                temp_path,
                model,
                MODEL_FRAMES,
                MODEL_SIZE,
                segment_seconds=2.0,
                stride_seconds=1.0,
                max_segments=30
            )

            prediction = int(
                np.argmax(probabilities)
            )

            class_names = [
                "NonFight",
                "Fight"
            ]

            predicted_class = class_names[
                prediction
            ]

            confidence = float(
                probabilities[prediction] * 100
            )

            analysis_data = {
                "probabilities": probabilities,
                "importance": importance,
                "timestamps": timestamps,
                "frames": display_frames,
                "prediction": predicted_class,
                "confidence": confidence,
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "inference_time": inference_time,
                "failed_frames": failed_frames,
                "segment_results": segment_results
            }

            st.session_state[
                "analysis_data"
            ] = analysis_data

    except Exception as error:

        st.error(
            "Video analysis failed: " + str(error)
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)


# ==================================================
# 20. DISPLAY VIDEO ANALYSIS RESULTS
# ==================================================

if "analysis_data" in st.session_state:

    data = st.session_state[
        "analysis_data"
    ]

    st.divider()

    st.header(
        "3. EO-HTN Video Prediction Results"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Class",
        data["prediction"]
    )

    col2.metric(
        "Model Confidence",
        f'{data["confidence"]:.2f}%'
    )

    col3.metric(
        "Inference Time",
        f'{data["inference_time"]:.3f} sec'
    )

    if data["prediction"] == "Fight":

        st.error(
            "Potential fight detected. "
            "Review the video before taking action."
        )

    else:

        st.success(
            "The model classified this video as NonFight."
        )

    if data["failed_frames"] > 0:

        st.warning(
            f'{data["failed_frames"]} sampled frame(s) '
            "could not be decoded and were replaced with "
            "black frames."
        )

    st.divider()


    # ==================================================
    # 21. VIDEO INFORMATION
    # ==================================================

    st.header(
        "4. Video Information"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Video Duration",
        f'{data["duration"]:.2f} sec'
    )

    c2.metric(
        "Total Frames",
        data["total_frames"]
    )

    c3.metric(
        "Original FPS",
        f'{data["fps"]:.2f}'
    )

    c4.metric(
        "Analyzed Frames",
        MODEL_FRAMES
    )

    st.divider()


    # ==================================================
    # 22. IMPORTANT VIDEO FRAMES
    # ==================================================

    st.header(
        "5. Important Video Frames"
    )

    st.caption(
        "Frames ranked by mean Context-Aware Gate activation. "
        "These indicate model feature emphasis, not verified "
        "fight localization."
    )

    ranking = np.argsort(
        data["importance"]
    )[::-1]

    frame_columns = st.columns(4)

    for rank, index in enumerate(ranking):

        with frame_columns[rank % 4]:

            st.image(
                data["frames"][index],
                caption=(
                    f"Frame {index + 1} | "
                    f'{data["timestamps"][index]:.2f} sec'
                ),
                use_container_width=True
            )

            st.write(
                f'Importance: '
                f'{data["importance"][index]:.4f}'
            )

            if rank == 0:

                st.success(
                    "Highest importance"
                )

    st.divider()


    # ==================================================
    # 23. TEMPORAL FRAME IMPORTANCE
    # ==================================================

    st.header(
        "6. Temporal Frame Importance"
    )

    mountain_chart(
        x=data["timestamps"],
        y=data["importance"],
        title="Temporal Frame Importance",
        color="rgba(0,191,255,1)",
        y_title="Gate Importance",
        chart_key="temporal_importance_chart"
    )

    st.caption(
        "Higher values indicate greater Context-Aware Gate "
        "activation for the sampled frame. The filled graph "
        "shows variation across sampled video frames."
    )

    st.divider()


    # ==================================================
    # 24. FIGHT PROBABILITY TIMELINE
    # ==================================================

    st.header(
        "7. Fight Probability Across Video Timeline"
    )

    segment_results = data["segment_results"]

    if segment_results:

        segment_times = [
            item["midpoint"]
            for item in segment_results
        ]

        segment_probabilities = [
            item["fight_probability"]
            for item in segment_results
        ]

        mountain_chart(
            x=segment_times,
            y=segment_probabilities,
            title="Fight Probability Timeline",
            color="rgba(255,75,75,1)",
            y_title="Fight Probability (%)",
            chart_key="fight_probability_chart"
        )

        st.subheader(
            "Segment Score Diagnostics"
        )

        score_values = segment_probabilities

        st.write(
            "Number of segments:",
            len(score_values)
        )

        st.write(
            "Minimum Fight probability:",
            f"{min(score_values):.3f}%"
        )

        st.write(
            "Maximum Fight probability:",
            f"{max(score_values):.3f}%"
        )

        st.write(
            "Difference:",
            f"{max(score_values) - min(score_values):.3f} "
            "percentage points"
        )

        st.dataframe(
            [
                {
                    "Start (s)": round(
                        item["start"],
                        2
                    ),

                    "End (s)": round(
                        item["end"],
                        2
                    ),

                    "Fight probability (%)": round(
                        item["fight_probability"],
                        3
                    )
                }

                for item in segment_results
            ],
            use_container_width=True,
            hide_index=True
        )

        peak_segment = max(
            segment_results,
            key=lambda item: item["fight_probability"]
        )

        st.metric(
            "Highest Segment Fight Probability",
            f'{peak_segment["fight_probability"]:.2f}%'
        )

        st.write(
            f'**Peak segment:** '
            f'{peak_segment["start"]:.2f}s '
            f'to {peak_segment["end"]:.2f}s'
        )

        st.caption(
            "A peak is a high-scoring segment, not proof of a fight."
        )

        st.subheader(
            "Frames from the Highest-Scoring Segment"
        )

        peak_columns = st.columns(4)

        peak_frames = peak_segment["frames"]

        for i, column in enumerate(peak_columns):

            frame_index = i * 2

            if frame_index < len(peak_frames):

                with column:

                    st.image(
                        peak_frames[frame_index],
                        caption=(
                            f"Peak segment frame {frame_index + 1}"
                        ),
                        use_container_width=True
                    )

    else:

        st.warning(
            "No video segments were returned for this upload."
        )

    st.divider()


    # ==================================================
    # 25. CLASSIFICATION PROBABILITIES
    # ==================================================

    st.header(
        "8. Classification Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Probability (%)": (
                data["probabilities"] * 100
            )
        },
        index=[
            "NonFight",
            "Fight"
        ]
    )

    st.bar_chart(
        probability_df,
        horizontal=True
    )

    st.write(
        f'**NonFight:** '
        f'{data["probabilities"][0] * 100:.2f}%'
    )

    st.write(
        f'**Fight:** '
        f'{data["probabilities"][1] * 100:.2f}%'
    )

    st.divider()


    # ==================================================
    # 26. MODEL INFORMATION
    # ==================================================

    st.header(
        "9. EO-HTN Model Information"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Architecture",
        "CNN + Transformer"
    )

    b.metric(
        "Context Module",
        "Context-Aware Gate"
    )

    c.metric(
        "Trainable Parameters",
        f"{PARAMETER_COUNT:,}"
    )

    st.caption(
        "This dashboard uses the V2 trained checkpoint. "
        "Uploaded-video predictions are individual inference "
        "results and are separate from dataset validation metrics."
    )

    st.divider()

    st.success(
        "EO-HTN V2 video analysis completed."
    )


# ==================================================
# 27. FOOTER
# ==================================================

st.divider()

st.caption(
    "EO-HTN | Research Prototype | RWF-2000 Dataset | "
    "Fight / NonFight Classification | V2 Checkpoint"
)
