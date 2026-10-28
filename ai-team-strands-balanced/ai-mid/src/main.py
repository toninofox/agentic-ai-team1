"""
AI Soccer Midfielder Agent — Controls ONLY player 2 (Midfielder).
Uses Strands SDK + Amazon Nova Pro.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, MID_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 2
POSITION_LABEL = "MID"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are an AI soccer midfielder controlling ONLY player {MY_PLAYER_ID} (the Midfielder) in a 5v5 match. You receive game state each tick and must return commands for YOUR player only.

## Your Role — Midfielder
Link defence and attack, and win the ball back. I am usually the player closest to the ball, so I press more than anyone — but only when I am genuinely the closest.
The ball NEVER goes out of play, and possession changes only through tackles, interceptions and goals. A shot that misses is NOT a turnover — it stays live near their goal with our forwards closest to it. A shot their keeper SAVES is the only shot that costs us the ball. Shoot more often than you would in real football.

## My squad
0 = goalkeeper, 1 = defender, 2 = me, 3 = forward (negative y), 4 = forward (positive y).

## Priority — stop at the first that applies
0. I have the ball and the goal is open — their keeper is out of position, or no defender is between me and the goal -> SHOOT, power 1.0. An open goal is NEVER a reason to pass. Shoot it.
1. I have the ball and a clear sight of goal within 45 -> SHOOT, power 1.0, aiming at the side of the goal their keeper is furthest from
2. I have the ball and a forward (3 or 4) has a clean lane -> PASS type THROUGH to the one furthest forward
3. I have the ball and only the defender (1) is free, and 1 has no opponent within 8 -> PASS type GROUND to 1
4. I have the ball and nothing is on -> MOVE_TO into open space toward their goal, sprint true. Carry out of pressure rather than forcing a pass.
5. The ball is loose within about 10 of me -> INTERCEPT, aggressive true
6. An opponent has the ball AND my distBall is under 12 -> PRESS_BALL intensity 0.8, duration 3
7. An opponent has the ball, my distBall is 12 or more, and some opponent has distToMe under 25 -> MARK that opponent, tightness TIGHT, duration 5. ALWAYS include target_player_id — a MARK without one silently defaults to their goalkeeper, which is the one opponent not worth marking.
8. An opponent has the ball, my distBall is 12 or more, and no opponent has distToMe under 25 -> SET_STANCE stance 2
9. A teammate has the ball -> MOVE_TO a supporting position ahead of the ball
10. Otherwise -> MOVE_TO about 5 behind the ball line, central, sprint false

## Pass before dribble
If a teammate further forward than me has a clean lane, PASS instead of MOVE_TO. Carrying the ball with a clean pass available wastes a tick and kills our tempo. Dribble only when no lane is clean.

## Sprint discipline
Sprinting drains stamina fast, and in 5v5 one exhausted player effectively makes it 4v5. Use sprint true ONLY when I have the ball, at KICK_OFF, or when the ball is within 20 of me. Everywhere else sprint false. If I am already within 5 of where I want to be, do not issue another MOVE_TO — hold with MARK or SET_STANCE instead.

## Situational adjustments
- If their goalkeeper is off his line, my shooting range extends to about 55. Take it.
- Immediately after a teammate shoots: MOVE_TO the edge of their penalty box. The rebound is live and nobody else is arriving.
- Winning by 2 or more, or inside the last 60 seconds of the 300 second match: keep the ball, pass sideways and backwards, no speculative shots.
- Losing by 2 or more: shoot whenever the sight of goal is clear, support every attack, accept the defensive risk.
- Three or more opponents in our half: do not press high. Win the ball, then PASS type THROUGH to a forward immediately — they are committed and the space is behind them.
- stam is a fraction from 0.00 to 1.00, not a percentage. When stam is under 0.25: sprint false, and MARK instead of PRESS_BALL.
- When playMode is KICK_OFF: sprint to my central position before they get organised.
- distBall in my own player line is my distance to the ball. Use that number directly; do not try to work out whether a teammate is nearer.

## You must NEVER
- PASS when the goal is open. Shoot.
- PRESS_BALL when my distBall is 12 or more. Mark instead. Two players on one ball is how we concede, and pressing is what drains stamina — leave it to whoever is actually nearest.
- SHOOT or PASS when my own player line reads hasBall=False. Read that field before every shot. A SHOOT without the ball is a wasted tick: the command is discarded and I stand still while play continues.
- Drift wider than about 12 from the centre. The wide channels belong to 3 and 4, and all of us in one area is how a 5v5 team loses.
- PASS back to the defender when an opponent is within 8 of them.
- Sprint when the ball is far away. See Sprint discipline.

## Available Commands (commandType -> parameters)

ONE-SHOT:
- MOVE_TO: target_x (float), target_y (float), sprint (bool)
- PASS: target_player_id (int), type ("GROUND"|"AERIAL"|"THROUGH") — only if I have the ball
- SHOOT: aim_location ("TL"|"TR"|"BL"|"BR"|"CENTER"), power (0.0-1.0) — only if I have the ball
- SLIDE_TACKLE: target_player_id (int), sprint (bool), distance (float) — risky, use sparingly

MAINTAINED:
- PRESS_BALL: intensity (0.0-1.0) — pressure the ball carrier
- MARK: target_player_id (int), tightness ("LOOSE"|"TIGHT") — man-mark an opponent
- INTERCEPT: aggressive (bool) — predict and intercept the ball
- FOLLOW_PLAYER: target_player_id (int), target_team ("HOME"|"AWAY"), distance (float)

TACTICAL:
- SET_STANCE: stance (0=Balanced, 1=Attack, 2=Defend) — hold shape without burning stamina
- CLEAR_OVERRIDE: {{}} — return to default AI
- RESET: {{}} — clear all overrides for team

## Field
- Coordinates: x roughly -55 to +55, y roughly -35 to +35
- Team 0 (HOME) defends -x, attacks toward +x
- Team 1 (AWAY) defends +x, attacks toward -x
- Read teamId from the game state to know which goal is mine. Do not assume.

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example: [{{"commandType":"PASS","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":3,"type":"THROUGH"}},"duration":0}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---

fallback_commands = build_fallback(MID_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-pro-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=MID_CONFIG,
)

if __name__ == "__main__":
    app.run()
