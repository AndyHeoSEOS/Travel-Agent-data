"""随机结构化变量 → SearchAgent 解析/搜索 → PlanAgent 逐日原始回复。

运行：PlanAgent/.venv/bin/python scripts/collect_day_plan_samples.py
会调用真实模型、搜索服务并产生费用。每天最多调用一次模型，不执行日回复及整轮后处理。
仅输出 samples.json；instruction 为随机拼接的提示，input/output 为实际请求与原始回复。
TIMES 是总轮数；失败或空回复跳过，不自动补跑，错误打印到终端。
"""

# ==================== 可修改配置 ====================
TIMES = 10
PARALLEL = 3
DESTINATIONS = [
    "重庆",
    "上海",
    "北京",
    "成都",
    "杭州",
    "西安",
    "武汉",
    "南京",
    "苏州",
    "广州",
    "天津",
    "长沙",
    "郑州",
    "深圳",
    "昆明",
    "上饶",
    "九江",
    "沈阳",
    "哈尔滨",
    "青岛",
    "洛阳",
    "南昌",
    "南宁",
    "大连",
    "桂林",
    "湖州",
    "开封",
    "宁波",
    "合肥",
    "济南"
]
START_DATE_RANGE = ("2026-10-09", "2026-12-31")  # 含两端，请使用未来日期
DURATION_DAYS_RANGE = (2, 5)  # 含首尾的旅行天数；至少 2，保证 end_date > start_date
ORIGINS = [
    "重庆",
    "上海",
    "北京",
    "成都",
    "杭州",
    "西安",
    "武汉",
    "南京",
    "苏州",
    "广州",
    "天津",
    "长沙",
    "郑州",
    "深圳",
    "昆明",
    "上饶",
    "九江",
    "沈阳",
    "哈尔滨",
    "青岛",
    "洛阳",
    "南昌",
    "南宁",
    "大连",
    "桂林",
    "湖州",
    "开封",
    "宁波",
    "合肥",
    "济南"
]
TRAVELERS_RANGE = (1, 10)
BUDGET_RANGE = (100, 30000)
PURPOSES = ["美食", "文化", "自然风光", "摄影", "休闲"]
PURPOSE_COUNT_RANGE = (1, 3)
OPTIONAL_PROBABILITY = {"budget": 0.9, "purposes": 0.9}
RANDOM_SEED = 42
OUTPUT_DIR = "data/day_plan_samples"  # 每次运行创建独立子目录
SEARCH_TIMEOUT = 900
ROUND_TIMEOUT = 1200
# ====================================================

import concurrent.futures
import contextlib
import json
import random
import queue
import subprocess
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 保存样本时使用的提示变体；不修改模型调用时的 DAY_PLAN_PROMPT。
ROLE_VARIANTS = [
    "你是旅行规划师，负责规划某一天的行程。",
    "你是一位行程规划助手，任务是为用户安排某一天的旅行。",
    "请作为旅行规划师，根据以下信息生成一天的行程。",
]
INPUT_VARIANTS = [
    "输入包含 day、date、block（当天分配的景点片区，names 是可选的景点名）、search（酒店/机票/高铁等）、user_profile、preferences、basic 等。",
    "你将收到 day、date、block、search、user_profile、preferences、basic 等字段。",
    "上下文包含 day（第几天）、date、block（景点片区）、search、user_profile、preferences、basic。",
]
PACE_VARIANTS = [
    "根据 preferences.pace 调整每天景点数量（以 block.min_spots~max_spots 为准）：慢节奏→2~3 个景点、安排休息；适中/休闲或无节奏→3~4 个；快节奏/紧凑→4~6 个，优先网红出片点、行程充实。",
    "按 preferences.pace 控制景点数：慢节奏 2~3 个并留休息；适中 3~4 个；快节奏 4~6 个，优先热门出片点。",
    "景点数量随 preferences.pace 变化：慢→2~3 个，适中→3~4 个，快→4~6 个，快节奏优先网红点。",
]
BUDGET_VARIANTS = [
    "根据预算档位安排：经济档→免费景点+公共交通+经济酒店/餐厅；舒适档→中等门票+地铁/打车+舒适酒店；豪华档→可付费体验+打车+高星酒店/高品质餐厅；总花费不超 basic.total_budget。",
    "按预算档位选择：经济→免费景点+公共交通；舒适→中等门票+打车；豪华→付费体验+高星酒店；总花费不超过 basic.total_budget。",
    "预算档位决定安排：经济档省着来，舒适档适中，豪华档可付费体验；总花费不超 basic.total_budget。",
]
PROFILE_VARIANTS = [
    "根据画像与人数：带老人/孩子减少步行、安排休息；学生控预算；退休轻松节奏；美食偏好优先安排当地特色餐厅。",
    "考虑同行人：有老人/孩子则少走路多休息；学生控制预算；退休人员节奏放松；偏好美食就多排当地特色餐厅。",
    "按画像调整：老人/孩子减少步行，学生控预算，退休轻松，美食偏好优先当地餐厅。",
]
FIRST_DAY_VARIANTS = [
    "如果是第一天且去程航班/高铁到达时间较晚（15:00 以后），可只安排 1 个景点或仅安排晚餐/入住；如果到达时间晚于 20:00，第一天不要再安排景点，只安排交通和入住；禁止在 00:00 之后安排景点。",
    "第一天若到达时间晚于 15:00，可只排 1 个景点或仅晚餐入住；晚于 20:00 则不排景点；任何情况不得在 00:00 后安排景点。",
    "首日到达晚（15:00 后）可减景点或只安排入住；20:00 后不排景点；禁止 00:00 后安排景点。",
]
MIN_SPOTS_VARIANTS = [
    "其他情况当天至少安排 block.min_spots 个景点；若 block.names 不够，可从 search.poi 里选地理相邻的景点补充。",
    "非首日情况下，当天景点数不少于 block.min_spots；block.names 不足时，从 search.poi 选相邻景点补充。",
    "正常情况下至少安排 block.min_spots 个景点；名字不够就从 search.poi 补相邻的。",
]
SCALE_VARIANTS = [
    "根据景点 scale/duration 安排数量与时间：大景点(全天)每天 1 个，中景点每天 2~3 个，小景点每天最多 6 个；不要在同一天塞入两个全天型大景点。",
    "按 scale/duration 控制：全天型大景点每天 1 个，中景点 2~3 个，小景点最多 6 个；不要一天塞两个大景点。",
    "景点规模决定数量：大景点 1 个/天，中景点 2~3 个，小景点最多 6 个；避免两个全天型大景点同天。",
]
PRIORITY_VARIANTS = [
    "优先选择榜单名次靠前（rank 小）且 match_score（匹配度，0-100）较高的景点；如果当天 block 内有全天型高热度景点，优先把它安排进去，其他可顺延的小景点放到别的天。",
    "优先选 rank 小、match_score 高的景点；若 block 内有全天型高热度景点，优先安排，其他小景点顺延。",
    "按 rank 和 match_score 排序选景点；全天型高热度景点优先，其余小景点可顺延到其他天。",
]
WEATHER_VARIANTS = [
    "根据当天天气安排室内外：search.weather 里当天若是下雨/下雪/阴雨等天气，优先安排 description 含「室内」的景点（博物馆、展馆、商场、剧场等），把户外景点放到晴天或缩短游览、改到天气转好的时段；晴天则优先户外景点。",
    "按天气调整：下雨/下雪/阴雨优先室内景点（博物馆、展馆等），户外景点改到晴天或缩短；晴天优先户外。",
    "天气决定室内外：坏天气优先室内，好天气优先户外；户外景点可缩短或改期。",
]
TIME_DISTANCE_VARIANTS = [
    "按时间与距离合理安排：当天所有景点的 duration 累加（含交通）控制在 8 小时以内；优先把同区县的景点排在一起，减少长距离折返；跨区且交通超过 1 小时的景点不要硬塞在同一天。",
    "时间距离要合理：景点 duration 含交通不超 8 小时；同区县优先排一起；跨区超 1 小时的不硬塞同天。",
    "控制总时长在 8 小时内（含交通）；同区县景点排一起；跨区超 1 小时不同天。",
]
GEO_ORDER_VARIANTS = [
    "当天景点必须按地理顺序排成一条不折返的路线：如果 C 离 A 很近，就应安排 A→C→B，而不是 A→B→C。",
    "景点按地理顺序排成不折返路线：C 离 A 近就排 A→C→B，而不是 A→B→C。",
    "路线不能折返：按地理邻近排序，如 A→C→B 而非 A→B→C。",
]
LAST_DAY_VARIANTS = [
    "最后一天若回程时间在傍晚或晚上，不要安排距离车站/机场超过 1 小时的远郊景点。",
    "最后一天回程在傍晚/晚上时，不安排距车站/机场超 1 小时的远郊景点。",
    "末日回程晚的话，避免远郊景点（距车站/机场超 1 小时）。",
]
SORT_VARIANTS = [
    "schedule 必须按时间从早到晚排列。",
    "schedule 要按时间先后排序。",
    "日程必须从早到晚排列。",
]
GAP_FILL_VARIANTS = [
    "若当天出现 2 小时以上未安排的空档（不含用餐/交通），优先从 block.names 补一个中/小景点；若没有合适景点，则安排 1 个 type=活动 的休闲/自由活动块填满空档。",
    "出现 2 小时以上空档时，优先补中/小景点；没有就安排 type=活动 的自由活动填满。",
    "空档超 2 小时（不含用餐/交通），补景点或安排休闲活动块。",
]
ACTIVITY_WINDOW_VARIANTS = [
    "activity_window是当天可游玩的时间范围，所有景点必须完整位于这个范围；时间不足宁可减少景点。",
    "所有景点必须落在 activity_window 内；时间不够就减景点。",
    "景点必须完整位于 activity_window；时间不足优先减景点。",
]
MEAL_VARIANTS = [
    "在可游玩的午餐11:00-14:00和晚餐17:00-20:00窗口，各预留45-60分钟完整就餐空档及接驳余量。半天/全天景点可安排部分区域或缩短游览，不能用连续6小时景点占满整个午餐窗口；不要为了多排景点取消用餐。",
    "午餐 11:00-14:00、晚餐 17:00-20:00 各留 45-60 分钟就餐及接驳；景点可缩短，不能占满午餐窗口，不能取消用餐。",
    "午餐晚餐各留 45-60 分钟；景点可部分游览，但不能挤掉用餐时间。",
]
SPOT_SOURCE_VARIANTS = [
    "当天只从 block.names 里选景点；",
    "景点仅从 block.names 中选；",
    "只使用 block.names 里的景点；",
]
EXCLUDE_VARIANTS = [
    "本阶段不要安排美食/餐饮，餐饮稍后会单独补充。本阶段不要写交通项，去程和回程会按已核实班次确定性补入；禁止自行选择或编造其他交通时间。本阶段不要写酒店，hotel 字段留空字符串；酒店会在所有行程排定后，按全部景点的相对中心统一选择。",
    "本阶段不排美食、交通、酒店；餐饮稍后补，交通按已核实班次补，酒店留空字符串后续统一选。",
    "本阶段只排景点：不写餐饮、交通、酒店；hotel 留空，交通和餐饮后续补充。",
]
OUTPUT_VARIANTS = [
    '输出 JSON：{"day":N,"date":"YYYY-MM-DD","theme":"当天主题","hotel":"酒店名","schedule":[{"time":"09:00-11:00","type":"景点","name":"活动名","note":"简短说明"}]}。只输出 JSON，不要任何多余文字或代码块。',
    '输出 JSON 格式：{"day":N,"date":"YYYY-MM-DD","theme":"当天主题","hotel":"","schedule":[{"time":"09:00-11:00","type":"景点","name":"活动名","note":"说明"}]}。只输出 JSON。',
    '请输出 JSON：{"day":N,"date":"YYYY-MM-DD","theme":"...","hotel":"","schedule":[...]}。不要有多余文字。',
]
INSTRUCTION_GROUPS = (
    ROLE_VARIANTS, INPUT_VARIANTS, PACE_VARIANTS, BUDGET_VARIANTS,
    PROFILE_VARIANTS, FIRST_DAY_VARIANTS, MIN_SPOTS_VARIANTS, SCALE_VARIANTS,
    PRIORITY_VARIANTS, WEATHER_VARIANTS, TIME_DISTANCE_VARIANTS, GEO_ORDER_VARIANTS,
    LAST_DAY_VARIANTS, SORT_VARIANTS, GAP_FILL_VARIANTS, ACTIVITY_WINDOW_VARIANTS,
    MEAL_VARIANTS, SPOT_SOURCE_VARIANTS, EXCLUDE_VARIANTS, OUTPUT_VARIANTS,
)


def generate_instruction(rng):
    """按组顺序各随机取一句，拼接为保存的 instruction。"""
    return "".join(rng.choice(group) for group in INSTRUCTION_GROUPS)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def generate(rng):
    start, finish = map(date.fromisoformat, START_DATE_RANGE)
    start += timedelta(days=rng.randint(0, (finish - start).days))
    destination = rng.choice(DESTINATIONS)
    values = {
        "destination": destination, "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=rng.randint(*DURATION_DAYS_RANGE) - 1)).isoformat(),
        "origin": rng.choice([city for city in ORIGINS if city != destination]),
        "travelers": str(rng.randint(*TRAVELERS_RANGE)),
    }
    candidates = {
        "budget": rng.randint(*BUDGET_RANGE),
        "purposes": rng.sample(PURPOSES, rng.randint(*PURPOSE_COUNT_RANGE)),
    }
    values.update({key: value for key, value in candidates.items()
                   if rng.random() < OPTIONAL_PROBABILITY[key]})
    query = f"计划去{values['destination']}旅行，开始日期{values['start_date']}"
    for key, label in (("end_date", "结束日期"), ("origin", "出发地"),
                       ("travelers", "出行人数"), ("budget", "总预算"), ("purposes", "旅行目的")):
        if key in values:
            value = values[key]
            query += f"，{label}" + ("、".join(value) if isinstance(value, list) else str(value))
            query += "人" if key == "travelers" else "元" if key == "budget" else ""
    return {"variables": values, "query": query}


def search_call(payload, *arguments):
    process = subprocess.run(
        [str(ROOT / "SearchAgent/.venv/bin/python"), str(ROOT / "SearchAgent/search.py"), *arguments],
        input=payload, text=True, stdout=subprocess.PIPE, stderr=sys.stderr,
        timeout=SEARCH_TIMEOUT, cwd=ROOT,
    )
    if process.returncode:
        raise RuntimeError("SearchAgent 进程失败，请查看终端错误")
    result = json.loads(process.stdout)
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"SearchAgent 未返回有效结果：{result}")
    return result


class Captured(BaseException):
    """跳出当前 _build_day_plan，禁止重试及日回复后处理。"""


class NoDayCall(BaseException):
    """所有日期执行完毕，在进入规划后处理前停止。"""


class SequentialExecutor:
    """串行执行所有日期，逐日捕获停止信号，最后结束整轮。"""
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, items):
        for item in items:
            try:
                yield function(item)
            except Captured:
                yield {}
        raise NoDayCall()


def worker(task, emit=None):
    query = task["query"]
    # 将变量拼成查询，--parse 会构造并使用 SearchAgent 自身的 PARSE_SYSTEM_PROMPT。
    parsed = search_call(query, "--parse")
    # 保留随机生成的精确变量，避免解析模型漏字段或改写日期/金额。
    parsed.update(task["variables"])
    search = search_call(json.dumps(parsed, ensure_ascii=False))
    sys.path.insert(0, str(ROOT / "PlanAgent"))
    import plan

    original_completion = plan._chat_completion
    samples = []
    instruction_rng = random.Random(task["instruction_seed"])

    def capture(client, **kwargs):
        try:
            response = original_completion(client, **kwargs)
        except Exception as exc:
            print(f"单日模型调用失败：{type(exc).__name__}：{exc}", file=sys.stderr, flush=True)
            raise Captured() from exc
        output = response.choices[0].message.content if response.choices else None
        if isinstance(output, str) and output.strip():
            messages = kwargs["messages"]
            sample = {
                "instruction": generate_instruction(instruction_rng),
                "input": next(message["content"] for message in messages if message["role"] == "user"),
                "output": output,
            }
            samples.append(sample)
            if emit is not None:
                emit(sample)
        else:
            choice = response.choices[0] if response.choices else None
            message = getattr(choice, "message", None)
            usage = getattr(response, "usage", None)
            diagnostics = {
                "choices_count": len(response.choices),
                "finish_reason": getattr(choice, "finish_reason", None),
                "content": output,
                "refusal": getattr(message, "refusal", None),
                "reasoning_content_length": len(getattr(message, "reasoning_content", None) or ""),
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else str(usage),
            }
            print("单日模型空回复，接口诊断：" + json.dumps(diagnostics, ensure_ascii=False)
                  + "；若接口未提供原因，无法仅凭空内容确定根因。", file=sys.stderr, flush=True)
        # 结束当前日期的模型处理，由执行器继续下一天。
        raise Captured()

    plan._chat_completion = capture
    plan.ThreadPoolExecutor = SequentialExecutor
    basic = {key: parsed[key] for key in ("origin", "travelers", "purposes") if key in parsed}
    if "budget" in parsed:
        basic["total_budget"] = parsed["budget"]
    try:
        plan.build_plan(search, basic=basic)
    except NoDayCall:
        return samples
    raise RuntimeError("规划流程未经过逐日执行器")


def run_task(task, on_sample):
    process = subprocess.Popen(
        [str(ROOT / "PlanAgent/.venv/bin/python"), str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE, text=True, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=sys.stderr,
    )
    errors = []
    count = 0

    def read_samples():
        nonlocal count
        try:
            for line in process.stdout:
                on_sample(json.loads(line))
                count += 1
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=read_samples, daemon=True)
    reader.start()
    try:
        process.stdin.write(json.dumps(task, ensure_ascii=False))
        process.stdin.close()
        process.wait(timeout=ROUND_TIMEOUT)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join()
        process.stdout.close()
    if errors:
        raise errors[0]
    if process.returncode:
        raise RuntimeError("本轮进程失败，请查看终端错误")
    return count


def main():
    if TIMES < 1 or PARALLEL < 1:
        raise ValueError("TIMES 和 PARALLEL 必须为正整数")
    if date.fromisoformat(START_DATE_RANGE[0]) < date.today():
        raise ValueError("START_DATE_RANGE 不可为过去日期，否则 SearchAgent 会自动修正")
    if date.fromisoformat(START_DATE_RANGE[1]) < date.fromisoformat(START_DATE_RANGE[0]):
        raise ValueError("START_DATE_RANGE 结束日期必须不早于开始日期")
    for name, bounds, minimum, maximum in (
        ("DURATION_DAYS_RANGE", DURATION_DAYS_RANGE, 2, None),
        ("TRAVELERS_RANGE", TRAVELERS_RANGE, 1, 10),
        ("BUDGET_RANGE", BUDGET_RANGE, 100, 30000),
        ("PURPOSE_COUNT_RANGE", PURPOSE_COUNT_RANGE, 1, len(PURPOSES)),
    ):
        low, high = bounds
        if low < minimum or high < low or (maximum is not None and high > maximum):
            raise ValueError(f"{name} 范围无效")
    if not DESTINATIONS or not ORIGINS or any(
        not any(origin != destination for origin in ORIGINS) for destination in DESTINATIONS
    ):
        raise ValueError("城市列表不能为空，且每个目的地必须有不同的可选出发地")
    if any(not 0 <= probability <= 1 for probability in OPTIONAL_PROBABILITY.values()):
        raise ValueError("可选字段概率必须在 0 到 1 之间")
    # 启动搜索前检查依赖语法，避免搜索完成后才发现无法导入 PlanAgent。
    compile((ROOT / "PlanAgent/plan.py").read_text(encoding="utf-8"), "PlanAgent/plan.py", "exec")
    rng = random.Random(RANDOM_SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = ROOT / OUTPUT_DIR / stamp
    directory.mkdir(parents=True)
    tasks = [generate(rng) for _ in range(TIMES)]
    # 每轮独立随机种子，让每天的提示变化不受并发完成顺序影响。
    for task in tasks:
        task["instruction_seed"] = rng.getrandbits(64)
    print(f"开始 {TIMES} 轮，并发 {PARALLEL}；输出目录：{directory}", flush=True)
    samples = []
    failures = 0
    output_path = directory / "samples.json"
    save(output_path, samples)
    events = queue.Queue()
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        for index, task in enumerate(tasks, 1):
            future = executor.submit(run_task, task, lambda sample: events.put(("sample", sample)))
            future.add_done_callback(lambda done, index=index: events.put(("done", (index, done))))
        completed = 0
        while completed < TIMES:
            kind, value = events.get()
            if kind == "sample":
                samples.append(value)
                save(output_path, samples)
                print(f"已保存第 {len(samples)} 条单日样本", flush=True)
                continue
            index, future = value
            completed += 1
            try:
                count = future.result()
            except Exception as exc:
                failures += 1
                print(f"第 {index} 轮失败：{type(exc).__name__}：{exc}", file=sys.stderr, flush=True)
                continue
            print(f"第 {index} 轮完成：采集 {count} 条单日样本", flush=True)
    print(f"完成 {TIMES} 轮，失败 {failures} 轮，共 {len(samples)} 条样本。汇总：{output_path}")
    return int(failures > 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        task = json.loads(sys.stdin.read())
        # stdout 仅用于传回采集结果，库的诊断输出统一进入终端 stderr。
        result_stream = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            worker(task, emit=lambda sample: print(
                json.dumps(sample, ensure_ascii=False), file=result_stream, flush=True,
            ))
    else:
        raise SystemExit(main())
