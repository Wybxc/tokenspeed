# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Check the real OpenAI-compatible HTTP endpoint after a clean server launch."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = []
    for prompt in ("What is 17 + 25? Answer briefly.", "用中文解释什么是递推状态。"):
        payload = dict(
            model="qwen35-08b",
            messages=[dict(role="user", content=prompt)],
            temperature=0,
            max_tokens=64,
            stream=False,
            chat_template_kwargs=dict(enable_thinking=False),
        )
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        cases.append(
            dict(
                prompt=prompt,
                wall_seconds=time.perf_counter() - started,
                response=result,
            )
        )
    first, second = [case["response"] for case in cases]
    assert first["choices"][0]["message"]["content"].strip() == "42", first
    assert first["choices"][0]["finish_reason"] == "stop", first
    assert first["usage"]["completion_tokens"] == 3, first
    assert second["usage"]["completion_tokens"] >= 32, second
    assert second["choices"][0]["message"]["content"].strip(), second
    output = dict(base_url=args.base_url, validated=True, cases=cases)
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
