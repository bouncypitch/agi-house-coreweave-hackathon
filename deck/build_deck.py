"""
Rebuilds deck_final.html from deck_template.html by inlining each video as a
base64 data: URI (the Artifact tool's CSP blocks loading external video
sources, so everything must be embedded directly in the page).

Run from the deck/ directory:  python3 build_deck.py
Produces deck_final.html (gitignored -- regenerate locally instead of
committing a multi-MB generated file).
"""

import base64

VIDEOS = {
    "loco_stage0": "../videos/locomotion/stage1_no_balance.mp4",
    "loco_stage1": "../videos/locomotion/stage2_learning_to_balance.mp4",
    "loco_stage2": "../videos/locomotion/stage3_few_steps.mp4",
    "loco_stage3": "../videos/locomotion/stage4_walking_forward.mp4",
    "loco_stage4": "../videos/locomotion/stage5_walking_backward.mp4",
    "catch_stage1": "../videos/catch/stage1_both_fall.mp4",
    "catch_stage2": "../videos/catch/stage2_balances_misses.mp4",
    "catch_stage3v1": "../videos/catch/stage3_catches_postcurriculum.mp4",
    "catch_stage4_goal": "../videos/catch/goal_target_behavior.mp4",
    "vision_disaster": "../videos/vision/disaster_response_concept.mp4",
}

with open("deck_template.html", "r") as f:
    html = f.read()

for key, path in VIDEOS.items():
    with open(path, "rb") as vf:
        data = vf.read()
    b64 = base64.b64encode(data).decode("ascii")
    uri = f"data:video/mp4;base64,{b64}"
    placeholder = "{{VIDEO:" + key + "}}"
    if placeholder not in html:
        print("WARNING - missing placeholder:", key)
    html = html.replace(placeholder, uri)

with open("deck_final.html", "w") as f:
    f.write(html)

print("wrote deck_final.html:", len(html), "bytes")
