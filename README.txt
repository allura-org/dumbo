▓█████▄  █    ██  ███▄ ▄███▓ ▄▄▄▄    ▒█████     __QQ  
▒██▀ ██▌ ██  ▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒  (_)_"> 
░██   █▌▓██  ▒██░▓██    ▓██░▒██▒ ▄██▒██░  ██▒ _)      
░▓█▄   ▌▓▓█  ░██░▒██    ▒██ ▒██░█▀  ▒██   ██░ version 0.0.1-rc
░▒████▓ ▒▒█████▓ ▒██▒   ░██▒░▓█  ▀█▓░ ████▓▒░ Research Preview, please steal
 ▒▒▓  ▒ ░▒▓▒ ▒ ▒ ░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░  
 ░ ▒  ▒ ░░▒░ ░ ░ ░  ░      ░▒░▒   ░   ░ ▒ ▒░  
 ░ ░  ░  ░░░ ░ ░ ░      ░    ░    ░ ░ ░ ░ ▒   
   ░       ░            ░    ░          ░ ░   
============================================================================
dumbo is a modular-by-design machine learning program, originally created
for training llms.

USAGE
`dumbo config.yml` runs the train described in config.yml. see `examples/`
for further (bad, vibe-configed) examples of configuration files.

LIMITATIONS / TODOS
- very unoptimized and some parts are untested (todo!!!)
    - needs flash_attn model patch(?)
    - needs cut cross entropy model patch
    - needs quantization/qlora patch
- single gpu only (for now)
- will only support fsdp even when multigpu is fully supported 
    (contributions welcome for deepspeed if you maintain it)
- no RL, online or offline (for now!)

ACKNOWLEDGEMENTS
allura <3
moonshot ai for creating kimi, the model that oneshot half of this codebase
anthropic for creating claude code, the harness that oneshot half of this
    codebase
