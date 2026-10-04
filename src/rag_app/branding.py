"""AgentCloud branding for the CLI: logo banner + prompt glyph."""
import sys

_C = "\033[96m"   # cyan
_O = "\033[38;5;214m"  # orange
_G = "\033[90m"   # grey
_B = "\033[1m"    # bold
_R = "\033[0m"    # reset

LOGO = rf"""{_C}
          .-~~~~~~-.
       .-~    __    ~-.
     .~      |  |       ~.        {_B}{_O}A G E N T   C L O U D   ·   A I{_R}{_C}
    (     .--|  |--.      )
    (    ( o )  ( o )     )       {_R}{_G}your private AWS solutions agent{_C}
     \    (  `--'  )     /        {_R}{_G}ask · design · visualize — grounded in AWS docs{_C}
      `~-._  `--'   _.-~
           `~------~'
{_R}"""

PROMPT = f"{_O}(~) agentcloud{_R} {_C}>{_R} "


def enable_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_utf8 = enable_utf8  # backwards-compatible alias


def print_banner() -> None:
    enable_utf8()
    print(LOGO)



def agent_say(text: str) -> str:
    return f"{_C}(~){_R} {text}"
