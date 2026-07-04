#!/usr/bin/env python3
from pyrosetta import *
from pyrosetta.rosetta.core.import_pose import pose_from_file
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
import argparse

def main():
    parser = argparse.ArgumentParser(description="Score protein-peptide complex with PyRosetta")
    parser.add_argument("-p", "--pdb", required=True, help="Input PDB file (complex with chains A,B receptor and C peptide)")
    parser.add_argument("-iface", "--interface", default="AC", help="Chains defining receptor vs peptide interface (default: AC)")
    args = parser.parse_args()

    # 初始化 PyRosetta
    init("-ex1 -ex2aro -use_input_sc")

    # 读入结构
    pose = pose_from_file(args.pdb)

    # ref2015 打分函数
    scorefxn = get_fa_scorefxn()

    # 1. 计算总分
    total_score = scorefxn(pose)
    print(f"Total ref2015 score: {total_score:.3f}")

    # 2. 计算界面 ΔΔG
    analyzer = InterfaceAnalyzerMover(args.interface, False, scorefxn)
    analyzer.apply(pose)
    ddg = analyzer.get_interface_dG()
    print(f"Interface ΔΔG ({args.interface}): {ddg:.3f}")

if __name__ == "__main__":
    main()