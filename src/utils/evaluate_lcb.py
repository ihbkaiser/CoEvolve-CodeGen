import io
import sys
import contextlib
import multiprocessing

def run_code(code, testtype, case, queue):
    """Helper function to run user code in a separate process."""
    try:
        if testtype == "stdin":
            input_data = case.get("input", "")
            if not input_data.endswith("\n"):
                input_data += "\n"
            f_in = io.StringIO(input_data)
            f_out = io.StringIO()
            f_err = io.StringIO()
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(f_out))
                stack.enter_context(contextlib.redirect_stderr(f_err))
                old_stdin = sys.stdin
                sys.stdin = f_in
                try:
                    exec_globals = {"__name__": "__main__"}
                    exec(code, exec_globals)
                finally:
                    sys.stdin = old_stdin
            queue.put((f_out.getvalue().strip(), f_err.getvalue().strip(), None))

        elif testtype == "functional":
            exec_globals = {}
            exec(code, exec_globals)
            if "main" not in exec_globals:
                raise RuntimeError("No function named 'main' found in code.")
            main_fn = exec_globals["main"]
            args = eval(case["input"])
            if not isinstance(args, tuple):
                args = (args,)
            output = str(main_fn(*args)).strip()
            queue.put((output, "", None))
        else:
            queue.put(("", "", ValueError(f"Unknown testtype: {testtype}")))
        
    except Exception as e:
        queue.put(("", "", e))

def evaluate(code: str, test_cases: list, timeout: int = 20):
    logs = []
    all_passed = True
    for i, case in enumerate(test_cases, start=1):
        testtype = case.get("testtype", "stdin")
        input_data = case.get("input", "")
        expected_output = str(case.get("output", "")).strip()
        passed = False
        output = ""
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_code, args=(code, testtype, case, queue))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            output = "Error: Timeout"
        else:
            try:
                out, err, exc = queue.get_nowait()
                if exc:
                    output = f"Error: {exc}"
                else:
                    output = (out + ("\n" if out and err else "") + err).strip()
            except Exception as e:
                output = f"Error: {e}"
        
        passed = (output == expected_output)
        if passed:
            logs.append(f"test {i} passed")
        else:
            log = f"""
test {i} failed,
input:
{input_data}
(expected:
{expected_output}
, got:
{output})"""
            logs.append(log)
            all_passed = False
    return "\n".join(logs), all_passed