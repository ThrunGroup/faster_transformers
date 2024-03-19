OUTFILE=final_methods_opt.csv
python profile_methods.py --ths 0.0 --filename $OUTFILE --cuths 0.0
python profile_methods.py --ths 0.005 --filename $OUTFILE --cuths 0.005
python profile_methods.py --ths 0.01 --filename $OUTFILE --cuths 0.01
python profile_methods.py --ths 0.03 --filename $OUTFILE --cuths 0.03
python profile_methods.py --ths 0.05 --filename $OUTFILE --cuths 0.05
python profile_methods.py --ths 0.1 --filename $OUTFILE --cuths 0.1
python profile_methods.py --ths 0.15 --filename $OUTFILE --cuths 0.15
python profile_methods.py --ths 0.2 --filename $OUTFILE --cuths 0.2
python profile_methods.py --ths 0.25 --filename $OUTFILE --cuths 0.25
python profile_methods.py --ths 0.3 --filename $OUTFILE --cuths 0.3
python profile_methods.py --ths 1.0 --filename $OUTFILE --cuths 1.0
python profile_methods.py --ths 2.0 --filename $OUTFILE --cuths 2.0
python profile_methods.py --ths 3.0 --filename $OUTFILE --cuths 3.0
python profile_methods.py --ths 4.5 --filename $OUTFILE --cuths 4.5