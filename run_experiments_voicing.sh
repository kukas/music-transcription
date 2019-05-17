
# python -u voicing_model.py --epochs 6 --architecture full_1layer --evaluate --logdir models/0406_011635-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-afull_1layer-cc0-b0-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --evaluate --logdir models/0406_072509-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b0-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --evaluate

# python -u voicing_model.py --epochs 6 --architecture full_1layer --first_pool_type avg --evaluate --logdir models/0406_014854-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptavg-fps1,5-fps1,5-cm8-afull_1layer-cc0-b0-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --first_pool_type avg --evaluate --logdir models/0406_075719-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptavg-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b0-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --first_pool_type avg --evaluate --logdir models/0406_082944-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptavg-fps1,5-fps1,5-cm8-anote_1layer-cc0-b0-d0.0-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --first_pool_type max --evaluate --logdir models/0406_022111-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptmax-fps1,5-fps1,5-cm8-afull_1layer-cc0-b0-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --first_pool_type max --evaluate --logdir models/0406_090737-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptmax-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b0-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --first_pool_type max --evaluate --logdir models/0406_093958-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptmax-fps1,5-fps1,5-cm8-anote_1layer-cc0-b0-d0.0-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --evaluate --logdir models/0406_025326-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-afull_1layer-cc0-b1-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --evaluate --logdir models/0406_101753-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b1-d0.0-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --evaluate --logdir models/0406_105045-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-anote_1layer-cc0-b1-d0.0-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --dropout 0.3 --evaluate --logdir models/0406_032609-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-afull_1layer-cc0-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --dropout 0.3 --evaluate --logdir models/0406_113355-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --dropout 0.3 --evaluate --logdir models/0406_120721-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-anote_1layer-cc0-b1-d0.3-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --dropout 0.3 --evaluate --logdir models/0406_035856-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-afull_1layer-cc0-b0-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --dropout 0.3 --evaluate --logdir models/0406_125404-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b0-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --dropout 0.3 --evaluate --logdir models/0406_132647-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-anote_1layer-cc0-b0-d0.3-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate --logdir models/0406_043123-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-afull_1layer-cc0-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate --logdir models/0406_140907-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-aoctave_1layer-cc0-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate --logdir models/0406_144342-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-anote_1layer-cc0-b1-d0.3-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate --logdir models/0406_050419-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm32-afull_1layer-cc0-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate --logdir models/0406_154413-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm32-aoctave_1layer-cc0-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate --logdir models/0406_162109-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm32-anote_1layer-cc0-b1-d0.3-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate --logdir models/0407_220612-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-afull_1layer-cc1-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate --logdir models/0407_223933-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-aoctave_1layer-cc1-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate --logdir models/0407_231420-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-anote_1layer-cc1-b1-d0.3-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate --logdir models/0408_001650-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-afull_1layer-cc2-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate --logdir models/0408_005047-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-aoctave_1layer-cc2-b1-d0.3-llconv-lcc0
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate --logdir models/0408_012536-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-anote_1layer-cc2-b1-d0.3-llconv-lcc0

# python -u voicing_model.py --epochs 6 --architecture full_1layer --last_conv_ctx 1 --evaluate --logdir models/0408_022625-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-afull_1layer-cc0-b0-d0.0-llconv-lcc1
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --last_conv_ctx 1 --evaluate --logdir models/0408_025837-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b0-d0.0-llconv-lcc1
# python -u voicing_model.py --epochs 6 --architecture note_1layer --last_conv_ctx 1 --evaluate --logdir models/0408_033128-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-anote_1layer-cc0-b0-d0.0-llconv-lcc1


# python -u voicing_model.py --epochs 6 --architecture full_1layer --last_conv_ctx 2 --evaluate --logdir models/0408_041510-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-afull_1layer-cc0-b0-d0.0-llconv-lcc2
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --last_conv_ctx 2 --evaluate --logdir models/0408_044739-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-aoctave_1layer-cc0-b0-d0.0-llconv-lcc2
# python -u voicing_model.py --epochs 6 --architecture note_1layer --last_conv_ctx 2 --evaluate --logdir models/0408_052035-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm8-anote_1layer-cc0-b0-d0.0-llconv-lcc2


# python -u voicing_model.py --epochs 6 --architecture full_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate --logdir models/0408_060834-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-afull_1layer-cc2-b1-d0.3-llconv-lcc2
# python -u voicing_model.py --epochs 6 --architecture octave_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate --logdir models/0408_064233-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-aoctave_1layer-cc2-b1-d0.3-llconv-lcc2
# python -u voicing_model.py --epochs 6 --architecture note_1layer --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate --logdir models/0408_071828-voicing-dmdb-bs32-s44100-hs1-fw512-cw2560-inTrue-apw10-mn24-nr72-bps5-as0.25-lr0.001-lrd1.0-lrds100000-cg0.0-llw0.0-mw1.0-scqt-fptNone-fps1,5-fps1,5-cm16-anote_1layer-cc2-b1-d0.3-llconv-lcc2


# python -u voicing_model.py --epochs 6 --architecture octave_octave --first_pool_type avg --evaluate
# python -u voicing_model.py --epochs 6 --architecture octave_octave --first_pool_type max --evaluate
# python -u voicing_model.py --epochs 6 --architecture octave_octave --batchnorm 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --last_conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate


# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --first_pool_type avg --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --first_pool_type max --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --last_conv_ctx 1 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate


# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --first_pool_type avg --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --first_pool_type max --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --last_conv_ctx 1 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate



# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --first_pool_type avg --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --first_pool_type max --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --last_conv_ctx 1 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_dilated --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate


# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --first_pool_type avg --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --first_pool_type max --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --last_conv_ctx 1 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture dilated_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate


# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --first_pool_type avg --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --first_pool_type max --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --last_conv_ctx 1 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate


# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --first_pool_type avg --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --first_pool_type max --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --dropout 0.3 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --last_conv_ctx 1 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --evaluate



# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --first_pool_type avg --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --first_pool_type max --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --last_conv_ctx 1 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate


# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --first_pool_type avg --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --first_pool_type max --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 32 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --last_conv_ctx 1 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture octave_note --batchnorm 1 --dropout 0.3 --capacity_multiplier 16 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate



# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_fix --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_fix --dropout 0.3 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_fix --dropout 0.3 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_fix --dropout 0.3 --conv_ctx 1 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_fix --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate

# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave --dropout 0.3 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave --dropout 0.3 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave --dropout 0.3 --conv_ctx 1 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate

# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave_octave --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave_octave --dropout 0.3 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave_octave --dropout 0.3 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave_octave --dropout 0.3 --conv_ctx 1 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_octave_octave --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate

# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave --dropout 0.3 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave --dropout 0.3 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave --dropout 0.3 --conv_ctx 1 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate

# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --dropout 0.3 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --dropout 0.3 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --dropout 0.3 --conv_ctx 1 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate


# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_octave_temporal --dropout 0.3 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_octave_temporal --dropout 0.3 --conv_ctx 1 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_octave_temporal --dropout 0.3 --conv_ctx 2 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_octave_temporal --dropout 0.3 --conv_ctx 1 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# # python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_octave_octave_temporal --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate


# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --capacity_multiplier 2 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --capacity_multiplier 4 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --capacity_multiplier 6 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
# python -u voicing_model.py --epochs 6 --evaluate_small_every 5000 --evaluate_every 10000 --architecture note_note_note_octave_octave --capacity_multiplier 10 --dropout 0.3 --conv_ctx 2 --last_conv_ctx 2 --harmonic_stacking 6 --evaluate
