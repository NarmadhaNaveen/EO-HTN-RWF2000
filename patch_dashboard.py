from pathlib import Path

path = Path("dashboard/app.py")
source = path.read_text(encoding="utf-8-sig")

# Add import
old = "from models.eo_htn import EOHTN"
new = old + "\nfrom utils.temporal_analysis import analyze_segments"

if new not in source:
    if old not in source:
        raise SystemExit("Could not find EOHTN import. No changes made.")
    source = source.replace(old, new, 1)

# Add segment inference after normal video inference
old = '''            ) = analyze_video(
                model, video_tensor
            )

            prediction = int('''

new = '''            ) = analyze_video(
                model, video_tensor
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

            prediction = int('''

if "segment_results = analyze_segments(" not in source:
    if old not in source:
        raise SystemExit("Could not find inference section. No changes made.")
    source = source.replace(old, new, 1)

# Store segment results
old = '''                "failed_frames": failed_frames
            }'''

new = '''                "failed_frames": failed_frames,
                "segment_results": segment_results
            }'''

if '"segment_results": segment_results' not in source:
    if old not in source:
        raise SystemExit("Could not find analysis result section. No changes made.")
    source = source.replace(old, new, 1)

# Add temporal probability chart after existing importance chart caption
old = '''    st.caption(
        "Higher values indicate greater Context-Aware Gate "
        "activation for the sampled frame."
    )

    # CLASS PROBABILITY GRAPH'''

new = '''    st.caption(
        "Higher values indicate greater Context-Aware Gate "
        "activation for the sampled frame."
    )

    # SEGMENT-LEVEL FIGHT PROBABILITY TIMELINE

    st.header("6. Fight Probability Across Video Timeline")

    segment_df = pd.DataFrame([
        {
            "Time (seconds)": item["midpoint"],
            "Fight probability (%)": item["fight_probability"]
        }
        for item in data["segment_results"]
    ]).set_index("Time (seconds)")

    st.line_chart(
        segment_df,
        y="Fight probability (%)",
        y_min=0,
        y_max=100,
        use_container_width=True
    )

    peak_segment = max(
        data["segment_results"],
        key=lambda item: item["fight_probability"]
    )

    st.metric(
        "Highest segment Fight probability",
        f'{peak_segment["fight_probability"]:.2f}%'
    )

    st.write(
        f'**Peak segment:** {peak_segment["start"]:.2f}s '
        f'to {peak_segment["end"]:.2f}s'
    )

    st.caption(
        "Each point is the actual Fight probability from the "
        "EO-HTN model applied to an overlapping video segment. "
        "A peak is a high-scoring segment, not proof of a fight."
    )

    st.subheader("Frames from the highest-scoring segment")

    peak_columns = st.columns(4)
    peak_frames = peak_segment["frames"]

    for i, column in enumerate(peak_columns):
        with column:
            st.image(
                peak_frames[i * 2],
                caption=f'Peak segment frame {i * 2 + 1}',
                use_container_width=True
            )

    # CLASS PROBABILITY GRAPH'''

if "Fight Probability Across Video Timeline" not in source:
    if old not in source:
        raise SystemExit("Could not find graph insertion point. No changes made.")
    source = source.replace(old, new, 1)

path.write_text(source, encoding="utf-8")
print("Dashboard updated successfully.")
