from colorama import Fore, Style, init

init(autoreset=True)

class Loger:

    def __init__(self, file=None):
        self.file = file
        self.args = "INFO"
        self._off = False

    @property
    def off(self):
        self._off = True
        return self

    @property
    def on(self):
        self._off = False
        return self


    def log(self, message, type="INFO"):
        if self.file is None:
            if type == "INFO":
                print(f"[{Style.BRIGHT}{Fore.GREEN}{type}] {message}{Style.RESET_ALL}")
            elif type == "ERROR":
                print(f"[{Style.BRIGHT}{Fore.RED}{type}] {message}{Style.RESET_ALL}")
            elif type == "WARNING":
                print(f"[{Style.BRIGHT}{Fore.YELLOW}{type}] {message}{Style.RESET_ALL}")
            else:
                print(f"[{Style.BRIGHT}{Fore.BLUE}{type}] {message}{Style.RESET_ALL}")
        else:
            with open(self.file, "a") as f:
                f.write(f"[{type}] {message}\n")

    def __call__(self, message):
        if self._off:
            return 
        return self.log(message, type=self.args)

    def __getitem__(self, item):
        self.args = item
        return self