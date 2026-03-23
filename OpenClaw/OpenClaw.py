import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from openai import OpenAI



class Config:
    MODEL = "gpt-4.1"
    TEMPERATURE = 0
    MAX_RETRIES = 2



@dataclass
class StepLog:
    step_id: int
    step: str
    output: str
    verified: bool
    retries: int
    timestamp: float



class Planner:
    def __init__(self, client: OpenAI):
        self.client = client

    def create_plan(self, task: str) -> List[str]:
        prompt = f"""
        Decompose the task into minimal, precise steps.

        Task:
        {task}

        Return a numbered list.
        """

        res = self.client.chat.completions.create(
            model=Config.MODEL,
            temperature=Config.TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )

        steps = res.choices[0].message.content.split("\n")
        return [s.strip() for s in steps if s.strip()]


class Executor:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, step: str) -> str:
        prompt = f"Execute this step precisely:\n{step}"

        res = self.client.chat.completions.create(
            model=Config.MODEL,
            temperature=Config.TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )

        return res.choices[0].message.content


class Verifier:
    def __init__(self, client: OpenAI):
        self.client = client

    def check(self, step: str, output: str) -> bool:
        prompt = f"""
        Step:
        {step}

        Output:
        {output}

        Is this correct? Answer YES or NO.
        """

        res = self.client.chat.completions.create(
            model=Config.MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return "YES" in res.choices[0].message.content.upper()



class OpenClawAgent:

    def __init__(self):
        self.client = OpenAI()
        self.planner = Planner(self.client)
        self.executor = Executor(self.client)
        self.verifier = Verifier(self.client)
        self.logs: List[StepLog] = []

    def run(self, task: str) -> Dict[str, Any]:
        steps = self.planner.create_plan(task)
        results = []

        for i, step in enumerate(steps):
            retries = 0

            while retries <= Config.MAX_RETRIES:
                output = self.executor.run(step)
                valid = self.verifier.check(step, output)

                if valid:
                    break

                retries += 1

            log = StepLog(
                step_id=i,
                step=step,
                output=output,
                verified=valid,
                retries=retries,
                timestamp=time.time()
            )

            self.logs.append(log)

            if not valid:
                print(f"[FAIL] Step {i} failed after retries.")
                break

            results.append(output)

        return {
            "task": task,
            "results": results,
            "logs": [asdict(log) for log in self.logs]
        }



if __name__ == "__main__":
    agent = OpenClawAgent()

    result = agent.run(
        "Analyze current challenges in Alzheimer's drug discovery"
    )

    print(json.dumps(result, indent=2))