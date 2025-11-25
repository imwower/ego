"""Teacher bridge to query Codex (CLI or API) when the SNN is confused/surprised.

The bridge formats internal SNN context into a mentoring-style system prompt.
Providers:
- codex_cli (default): call local `codex` command line tool if available.
- codex_api: use OpenAI chat API if the package/key are available.
Otherwise, a deterministic mock reply keeps tests offline.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class TeacherBridge:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        use_mock: bool = False,
        provider: str = "codex_cli",
        codex_cmd: str = "codex",
        codex_args: Optional[list] = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")
        self.codex_cmd = codex_cmd
        # Default to non-interactive exec; callers can override args.
        self.codex_args = codex_args or ["exec"]

        self._client = None
        self._codex_available = shutil.which(self.codex_cmd) is not None

        if self.provider == "codex_cli":
            self.use_mock = use_mock or not self._codex_available
        elif self.provider == "codex_api":
            self.use_mock = use_mock or self.api_key is None or OpenAI is None
            if not self.use_mock and OpenAI is not None:
                self._client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError("Unknown provider; use 'codex_cli' or 'codex_api'.")

    def ask_gemini(self, context_data: Dict[str, Any], trigger_type: str) -> Dict[str, Any]:
        """Query Codex (or mock) with a prompt built from SNN context.

        Args:
            context_data: dictionary with keys like
                - prediction_error_norm: float
                - recent_spikes: dict or description of active neurons
                - proto_state: dict with energy/pain/curiosity
                - notes: free-form string
            trigger_type: "CONFUSION" or "SURPRISE"

        Returns:
            dict containing the composed prompt and the reply text.
        """

        trigger = trigger_type.upper()
        if trigger not in {"CONFUSION", "SURPRISE"}:
            raise ValueError("trigger_type must be CONFUSION or SURPRISE")

        prompt = self._build_prompt(context_data, trigger)

        reply: str
        provider_ok = True
        raw_output = ""
        if self.use_mock:
            reply = self._mock_reply(context_data, trigger)
        elif self.provider == "codex_cli":
            reply, ok, raw_output = self._call_codex_cli(prompt)
            provider_ok = ok
            if not ok:
                reply = self._mock_reply(context_data, trigger)
        else:  # codex_api
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Provide a concise teaching tip (<=80 words) to reduce entropy.",
                    },
                ],
            )
            reply = response.choices[0].message.content

            raw_output = reply

        return {
            "prompt": prompt,
            "reply": reply,
            "provider": self.provider,
            "provider_ok": provider_ok,
            "raw_output": raw_output,
        }

    def _build_prompt(self, ctx: Dict[str, Any], trigger: str) -> str:
        pe = ctx.get("prediction_error_norm", "unknown")
        proto = ctx.get("proto_state", {}) or {}
        spikes = ctx.get("recent_spikes", {}) or {}
        notes = ctx.get("notes", "")

        proto_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in proto.items()
        ) or "energy=unknown, pain=unknown, curiosity=unknown"

        spike_str = " | ".join(f"{k}:{v}" for k, v in spikes.items()) or "none"

        system_prompt = f"""
You are Codex acting as a mentor/teacher for a neuromorphic SNN (Ego-Sphere).
Trigger: {trigger}
Prediction error norm: {pe}
Proto-self: {proto_str}
Recent spikes: {spike_str}
Context notes: {notes}

Responsibilities:
- Interpret the surprise/confusion signal.
- Provide a concise teaching tip (<=80 words) to reduce entropy.
- Suggest a simple rule or pattern the SNN can integrate.
- Keep tone instructional and concrete (no fluff).
"""
        return system_prompt.strip()

    def _mock_reply(self, ctx: Dict[str, Any], trigger: str) -> str:
        pe = ctx.get("prediction_error_norm", "?")
        proto = ctx.get("proto_state", {})
        pain = proto.get("pain", "?") if isinstance(proto, dict) else "?"
        return (
            f"[MOCK {trigger}] Detected high error={pe}. "
            f"If pain={pain}, damp irrelevant spikes; focus on recent pattern and predict its next token."
        )

    def _call_codex_cli(self, prompt: str) -> tuple[str, bool, str]:
        """Invoke local codex CLI; report success flag and raw output."""

        try:
            cmd = [self.codex_cmd]
            if self.codex_args:
                cmd.extend(self.codex_args)
            # Pass prompt as a positional argument (non-interactive), e.g.
            # codex exec --model gpt-4o-mini "<prompt>"
            cmd.append(prompt)
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=True,
            )
            output = result.stdout.strip()
            return (output or "[CODEX_CLI] Empty response", True, result.stderr.strip())
        except Exception as exc:  # pragma: no cover - environment dependent
            err = getattr(exc, "stderr", "") or str(exc)
            return (err, False, err)


__all__ = ["TeacherBridge"]
