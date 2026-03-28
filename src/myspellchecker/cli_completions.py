def generate_completion_script(shell: str) -> str:  # noqa: E501
    """Generate shell completion script.

    Args:
        shell: Shell type ('bash', 'zsh', 'fish')

    Returns:
        Shell completion script content

    Note: Line length warnings are suppressed as shell scripts have their own formatting.
    """
    if shell == "bash":
        return """# myspellchecker bash completion
_myspellchecker_completion() {
    local cur prev commands opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    commands="check build train-model completion"

    if [[ ${COMP_CWORD} == 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        return 0
    fi

    case "${COMP_WORDS[1]}" in
        check)
            opts="--output --format --level --db --no-phonetic --no-context \\
--no-ner --ner-model --ner-device --preset --verbose --color --no-color --help"
            case "${prev}" in
                --format|-f)
                    COMPREPLY=( $(compgen -W "json text csv rich" -- ${cur}) )
                    return 0
                    ;;
                --level)
                    COMPREPLY=( $(compgen -W "syllable word" -- ${cur}) )
                    return 0
                    ;;
                --preset|-p)
                    COMPREPLY=( $(compgen -W "default fast accurate minimal strict" -- ${cur}) )
                    return 0
                    ;;
                --db|--output|-o)
                    COMPREPLY=( $(compgen -f -- ${cur}) )
                    return 0
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) $(compgen -f -- ${cur}) )
            ;;
        build)
            opts="--input --output --work-dir --keep-intermediate --sample --col \\
--json-key --incremental --word-engine --seg-model --seg-device --validate \\
--worker-timeout --no-dedup --no-desegment --help"
            case "${prev}" in
                --word-engine)
                    COMPREPLY=( $(compgen -W "crf myword transformer" -- ${cur}) )
                    return 0
                    ;;
                --input|-i|--output|-o|--work-dir)
                    COMPREPLY=( $(compgen -f -- ${cur}) )
                    return 0
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            ;;
        train-model)
            opts="--input --output --architecture --epochs --batch-size --learning-rate \\
--warmup-ratio --weight-decay --hidden-size --layers --heads --max-length \\
--vocab-size --min-frequency --resume --keep-checkpoints --no-metrics --help"
            case "${prev}" in
                --input|-i|--output|-o|--resume)
                    COMPREPLY=( $(compgen -f -- ${cur}) )
                    return 0
                    ;;
                --architecture|-a)
                    COMPREPLY=( $(compgen -W "roberta bert" -- ${cur}) )
                    return 0
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            ;;
        completion)
            opts="--shell --help"
            case "${prev}" in
                --shell)
                    COMPREPLY=( $(compgen -W "bash zsh fish" -- ${cur}) )
                    return 0
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            ;;
    esac
}
complete -F _myspellchecker_completion myspellchecker
"""
    elif shell == "zsh":
        return """#compdef myspellchecker
# myspellchecker zsh completion

_myspellchecker() {
    local -a commands
    commands=(
        'check:Check text for spelling errors'
        'build:Build dictionary database from corpus'
        'train-model:Train a custom semantic model'
        'completion:Generate shell completion script'
    )

    _arguments -C \\
        '1: :->command' \\
        '*: :->args'

    case "$state" in
        command)
            _describe 'command' commands
            ;;
        args)
            case "$words[2]" in
                check)
                    _arguments \\
                        '(-o --output)'{-o,--output}'[Output file path]:file:_files' \\
                        '(-f --format)'{-f,--format}'[Output format]:format:(json text csv rich)' \\
                        '--level[Validation level]:level:(syllable word)' \\
                        '--db[Custom database path]:file:_files' \\
                        '--no-phonetic[Disable phonetic matching]' \\
                        '--no-context[Disable context checking]' \\
                        '--no-ner[Disable Named Entity Recognition]' \\
                        '--ner-model[HuggingFace NER model name]:model:' \\
                        '--ner-device[NER device (-1=CPU, 0+=GPU)]:device:' \\
                        '(-p --preset)'{-p,--preset}'[Configuration preset]:preset:' \\
                        '(default fast accurate minimal strict)' \\
                        '(-v --verbose)'{-v,--verbose}'[Enable verbose logging]' \\
                        '--color[Force color output]' \\
                        '--no-color[Disable color output]' \\
                        '*:input file:_files'
                    ;;
                build)
                    _arguments \\
                        '(-i --input)'{-i,--input}'[Input corpus files]:file:_files' \\
                        '(-o --output)'{-o,--output}'[Output database path]:file:_files' \\
                        '--work-dir[Directory for intermediate files]:directory:_directories' \\
                        '--keep-intermediate[Keep intermediate files]' \\
                        '--sample[Generate sample corpus]' \\
                        '--col[Column name for CSV/TSV]:column:' \\
                        '--json-key[Key name for JSON objects]:key:' \\
                        '--incremental[Incremental update]' \\
                        '--word-engine[Word segmentation engine]:engine:(crf myword transformer)' \\
                        '--seg-model[Custom model for transformer engine]:model:' \\
                        '--seg-device[Device for transformer inference]:device:' \\
                        '--validate[Validate inputs only]' \\
                        '--worker-timeout[Worker timeout in seconds]:timeout:' \\
                        '--no-dedup[Disable line deduplication]' \\
                        '--no-desegment[Keep word segmentation markers]'
                    ;;
                train-model)
                    _arguments \\
                        '(-i --input)'{-i,--input}'[Input corpus file]:file:_files' \\
                        '(-o --output)'{-o,--output}'[Output directory]:directory:_directories' \\
                        '(-a --architecture)'{-a,--architecture}'[Arch]:a:(roberta bert)' \\
                        '--epochs[Training epochs]:epochs:' \\
                        '--batch-size[Batch size]:size:' \\
                        '--learning-rate[Peak learning rate]:rate:' \\
                        '--warmup-ratio[LR warmup ratio]:ratio:' \\
                        '--weight-decay[Weight decay]:decay:' \\
                        '--hidden-size[Hidden layer size]:size:' \\
                        '--layers[Transformer layers]:layers:' \\
                        '--heads[Attention heads]:heads:' \\
                        '--max-length[Max sequence length]:length:' \\
                        '--vocab-size[Vocabulary size]:size:' \\
                        '--min-frequency[Min token frequency]:freq:' \\
                        '--resume[Resume from checkpoint]:directory:_directories' \\
                        '--keep-checkpoints[Keep checkpoints]' \\
                        '--no-metrics[Disable metrics saving]'
                    ;;
                completion)
                    _arguments \\
                        '--shell[Shell type]:shell:(bash zsh fish)'
                    ;;
            esac
            ;;
    esac
}

_myspellchecker "$@"
"""
    elif shell == "fish":
        return """# myspellchecker fish completion

# Disable file completion by default
complete -c myspellchecker -f

# Commands
complete -c myspellchecker -n "__fish_use_subcommand" -a check -d "Check text for spelling errors"
complete -c myspellchecker -n "__fish_use_subcommand" -a build \\
    -d "Build dictionary database from corpus"
complete -c myspellchecker -n "__fish_use_subcommand" -a train-model \\
    -d "Train a custom semantic model"
complete -c myspellchecker -n "__fish_use_subcommand" -a completion \\
    -d "Generate shell completion script"

# check command options
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -s o -l output \\
    -d "Output file path" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -s f -l format \\
    -d "Output format" -xa "json text csv rich"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l level \\
    -d "Validation level" -xa "syllable word"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l db \\
    -d "Custom database path" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l no-phonetic \\
    -d "Disable phonetic matching"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l no-context \\
    -d "Disable context checking"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l no-ner \\
    -d "Disable Named Entity Recognition"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l ner-model \\
    -d "HuggingFace NER model name"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l ner-device \\
    -d "NER device (-1=CPU, 0+=GPU)"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -s p -l preset \\
    -d "Configuration preset" -xa "default fast accurate minimal strict"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -s v -l verbose \\
    -d "Enable verbose logging"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l color -d "Force color output"
complete -c myspellchecker -n "__fish_seen_subcommand_from check" -l no-color \\
    -d "Disable color output"

# build command options
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -s i -l input \\
    -d "Input corpus files" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -s o -l output \\
    -d "Output database path" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l work-dir \\
    -d "Directory for intermediate files" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l keep-intermediate \\
    -d "Keep intermediate files"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l sample \\
    -d "Generate sample corpus"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l col \\
    -d "Column name for CSV/TSV"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l json-key \\
    -d "Key name for JSON objects"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l incremental \\
    -d "Incremental update"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l word-engine \\
    -d "Word segmentation engine" -xa "crf myword transformer"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l seg-model \\
    -d "Custom model name for transformer engine"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l seg-device \\
    -d "Device for transformer inference (-1=CPU, 0+=GPU)"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l validate \\
    -d "Validate inputs only"
complete -c myspellchecker -n "__fish_seen_subcommand_from build" -l worker-timeout \\
    -d "Worker timeout in seconds"

# train-model command options
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -s i -l input \\
    -d "Input corpus file" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -s o -l output \\
    -d "Output directory" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -s a -l architecture \\
    -d "Model architecture" -xa "roberta bert"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l epochs \\
    -d "Training epochs"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l batch-size \\
    -d "Batch size"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l learning-rate \\
    -d "Peak learning rate"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l warmup-ratio \\
    -d "LR warmup ratio"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l weight-decay \\
    -d "Weight decay"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l hidden-size \\
    -d "Hidden layer size"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l layers \\
    -d "Transformer layers"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l heads \\
    -d "Attention heads"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l max-length \\
    -d "Max sequence length"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l vocab-size \\
    -d "Vocabulary size"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l min-frequency \\
    -d "Min token frequency"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l resume \\
    -d "Resume from checkpoint" -r
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l keep-checkpoints \\
    -d "Keep checkpoints"
complete -c myspellchecker -n "__fish_seen_subcommand_from train-model" -l no-metrics \\
    -d "Disable metrics saving"

# completion command options
complete -c myspellchecker -n "__fish_seen_subcommand_from completion" -l shell \\
    -d "Shell type" -xa "bash zsh fish"
"""
    else:
        raise ValueError(f"Unsupported shell: {shell}. Use bash, zsh, or fish.")
