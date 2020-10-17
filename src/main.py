from src import Cube

def main():
    cube = Cube.Cube(verbose=False)
    # F2B + M slice centers
    cube.do_moves('y z2')  # yellow top, red front
    centers = ['U', 'R', 'F', 'D', 'L', 'B']
    F2B = ['LF', 'LD', 'LB', 'LFD', 'LBD', 'RF', 'RD', 'RB', 'RFD', 'RBD']
    for piece_str in centers + F2B:
        cube.set_piece_color([char for char in piece_str], 't')

    initial_state = cube.get_state()  # save the state to come back to it later

    eo_cases = ["M'U'MU2M'UM",
                "M'U'MU2M'UMU",
                "M'U'MU2M'UMU'",
                "M'U'MU2M'UMU2",
                "M'UMU'M'UM",
                "M'UMU'M'UMU",
                "M'U'MU'M'U'M",
                "M'U'MU'M'U'MU",
                "M'U'MU'M'U'MU'",
                "M'U'MU'M'U'MU2",
                "MU'MU2MU2M",
                "MUM",
                "MUMU",
                "MUMU'",
                "MUMU2"]

    cmll = "r U' r2' D' r U' r' D r2 U r'"
    cmll_setup = "L' U R U' L U' R' U' R U' R' M2 U M' U2 M U M2 U'"
    i = 0
    for eo_setup in eo_cases:
        url_before, url_after = cube.CMLL_affects_EO(cmll, cmll_setup, eo_setup)
        print(i, '\tBefore', url_before)
        print(' \tAfter ', url_after)
        cube.set_state(initial_state)
        i += 1


if __name__ == "__main__":
    main()
