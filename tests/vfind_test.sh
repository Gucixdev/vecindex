#!/usr/bin/env bash
# vfind_test.sh - vfind driven in a tmux pane of a set size: typing narrows the hits (the last word as a prefix),
# Enter opens the document, Enter there prints the choice on stdout; Esc quits with nothing chosen (exit 1).
# Needs tmux: skipped without it.
#   tests/vfind_test.sh .build/vfind .build/vfind_demo
set -u
vf=$1 demo=$2
command -v tmux >/dev/null || { echo "vfind: tmux missing - skipped"; exit 0; }
dir=$(mktemp -d /tmp/vfind-test.XXXXXX) || exit 1
sock=vfind-test-$$
fail=0
check() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; fail=1; fi; }
screen() { tmux -L "$sock" capture-pane -p -t vf; }
# wait up to 5 s for the screen to show $1
until_shows() { local i; for (( i = 0; i < 100; ++i )); do screen | grep -q -- "$1" && return 0; sleep 0.05; done; return 1; }
run() { tmux -L "$sock" new-session -d -s vf -x 100 -y 30 "NO_COLOR=1 $vf $dir/d.vxi > $dir/out; echo \$? > $dir/rc"; }
done_rc() { local i; for (( i = 0; i < 100; ++i )); do [[ -s $dir/rc ]] && { cat "$dir/rc"; return; }; sleep 0.05; done; echo timeout; }

echo "vfind (tmux, 100x30)"
"$demo" "$dir/d.vxi" || { echo "  FAIL the demo store"; exit 1; }
run
check "starts with the prompt and a hint" "until_shows 'type to search the texts'"
tmux -L "$sock" send-keys -t vf -l "zebr"
check "the word being typed is a prefix: zebr finds 3 chunks" "until_shows '3 hits'"
tmux -L "$sock" send-keys -t vf -l "afish"
check "zebrafish: 2 hits" "until_shows '2 hits'"
check "a hit: its document, chunk and id" "until_shows 'doc 2  chunk 1  id 2097153'"
check "a hit: a snippet with the word" "screen | grep -q 'zebrafish genome was sequenced'"
tmux -L "$sock" send-keys -t vf -l "x"
check "zebrafishx: no hits" "until_shows '0 hits'"
tmux -L "$sock" send-keys -t vf BSpace
check "Backspace: the hits are back" "until_shows '2 hits'"
tmux -L "$sock" send-keys -t vf -l " genome"
check "zebrafish genome: the chunk with both words first" "until_shows '2 hits' && screen | sed -n 2p | grep -q 'doc 2  chunk 1'"
tmux -L "$sock" send-keys -t vf Enter
check "Enter: the document, all of its chunks" "until_shows 'doc 2  3 chunks'"
check "the document: the hit's chunk is marked" "screen | grep -q 'chunk 1 (the hit)'"
check "the document: a newline inside a chunk starts a line" "screen | grep -q 'it has 26,000 genes'"
tmux -L "$sock" send-keys -t vf Escape
check "Esc: back to the list" "until_shows '2 hits'"
tmux -L "$sock" send-keys -t vf Enter
until_shows 'doc 2  3 chunks' >/dev/null
tmux -L "$sock" send-keys -t vf Enter
rc=$(done_rc)
check "Enter in the document: exit 0" "[[ $rc == 0 ]]"
check "stdout: ID<TAB>DOCUMENT<TAB>CHUNK" "[[ \$(cat $dir/out) == \$(printf '2097153\t2\t1') ]]"

rm -f "$dir/rc" "$dir/out"
run
until_shows 'type to search' >/dev/null
tmux -L "$sock" send-keys -t vf -l "cat"
check "cat: the cat document's chunks" "until_shows '4 hits'"
tmux -L "$sock" send-keys -t vf C-u
check "Ctrl-U clears the query" "until_shows 'type to search'"
tmux -L "$sock" send-keys -t vf Escape
rc=$(done_rc)
check "Esc in the list: exit 1, nothing on stdout" "[[ $rc == 1 && ! -s $dir/out ]]"

tmux -L "$sock" kill-server 2>/dev/null
"$vf" -h | grep -q 'ID<TAB>DOCUMENT<TAB>CHUNK' && echo "  ok   -h prints the help" || { echo "  FAIL -h"; fail=1; }
"$vf" 2>/dev/null; check "no store: exit 2" "[[ $? == 2 ]]"
echo "vfind: $([[ $fail == 0 ]] && echo OK || echo FAILED)"
exit $fail
