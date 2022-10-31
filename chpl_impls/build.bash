if [[ $# -eq 0 ]]; then
    echo "building chpl source ..."
    chpl nsStep11.chpl -o ./build/nsStep11 -screatePlots=true
    chpl nsStep12.chpl -o ./build/nsStep12 -screatePlots=true
else
    echo "building chpl source with --fast ..."
    chpl nsStep11.chpl -o ./release/nsStep11 -screatePlots=false --fast
    chpl nsStep12.chpl -o ./release/nsStep12 -screatePlots=false -stermOnTol=false --fast
fi
