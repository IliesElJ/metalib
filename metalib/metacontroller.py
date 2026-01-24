from multiprocessing import Process
from metalib.metaworker import run_strategy_loop
from metalib.utils import clean_args


class MetaController:
    def __init__(self):
        self.processes = {}  # script_name -> subprocess.Popen object

    def start_script(self, strategy_type, init_args):
        tag = init_args.get("tag")
        init_args = clean_args(init_args)

        # Start the strategy in its own process
        p = Process(target=run_strategy_loop, args=(strategy_type, init_args))
        p.start()
        self.processes[tag] = p

        # process = subprocess.Popen(["python", path])
        try:
            p = Process(target=run_strategy_loop, args=(strategy_type, init_args))
            message = f"Started {strategy_type} instance with PID {p.pid} and tag {tag}"
            pid = p.pid
            running = True
        except Exception as e:
            message = f"Error starting {strategy_type}: {e}"
            pid = None
            running = False

        return message, pid, running

    def stop_instance(self, tag):
        process = self.processes.get(tag)
        if not process:
            print(f"{tag} is not running.")
            return

        process.terminate()
        process.wait()
        del self.processes[tag]
        print(f"Stopped {tag}")

    def stop_all_running(self):
        for tag in self.list_processes():
            self.stop_instance(tag)
        return

    def list_processes(self):
        return list(self.processes.keys())
