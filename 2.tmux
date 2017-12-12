set -g mode-mouse on
set -g mouse-resize-pane on
set -g mouse-select-pane on
set -g mouse-select-window on
set-window-option -g mode-keys vi

set -g default-terminal "screen"
bind -t vi-copy y copy-pipe 'xclip -in -selection clipboard'
