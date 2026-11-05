"""Write the water dielectric table read by ccres_disdrometer_processing.dielectric.

One entry per radar frequency of the network : refractive index of water from
Ellison (2007) at the reference temperature, and |K|^2 of the band. Run it again
when a radar with a new frequency joins (then rebuild the scattering tables with
scripts/build_scattering_tables.py) :

    python scripts/build_dielectric_table.py 9.6 35.0 35.1 ...
"""

import argparse
import datetime

from ccres_disdrometer_processing import dielectric

# frequencies (GHz) written by Cloudnet in the radar files of the network
NETWORK_FREQUENCIES = (9.6, 35.0, 35.1, 35.15, 35.46, 35.5, 94.0, 95.0)


def main():
    """Write the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("frequency_ghz", nargs="*", type=float)
    parser.add_argument("--output", default=dielectric.TABLE_FILE)
    args = parser.parse_args()
    freqs = sorted(set(args.frequency_ghz or NETWORK_FREQUENCIES))

    lines = [
        "# Water properties for the scattering at each radar frequency of the",
        "# network, written by scripts/build_dielectric_table.py : do not edit.",
        "# refractive_index : Ellison (2007), J. Phys. Chem. Ref. Data 36, 1,",
        "# at the temperature below. kw_sqr : |K|^2 the radars of the band use",
        "# to compute Ze (X 0.93, Ka 0.88, W 0.74).",
        f'date_created = "{datetime.date.today().isoformat()}"',
        f"temperature = {dielectric.T_REF}",
        "",
    ]
    for f in freqs:
        n = dielectric.ellison_refractive_index(f * 1e9, dielectric.T_REF)
        kw = dielectric.kw_sqr_convention(f * 1e9)
        lines += [
            "[[entry]]",
            f"frequency_ghz = {f}",
            f"refractive_index = [{n.real:.6f}, {n.imag:.6f}]",
            f"kw_sqr = {kw}",
            "",
        ]
        print(f"{f:6.2f} GHz  n = {n.real:.4f} + {n.imag:.4f}i  |K|^2 = {kw}")
    with open(args.output, "w") as fh:
        fh.write("\n".join(lines))
    print(f"{len(freqs)} entries written to {args.output}")


if __name__ == "__main__":
    main()
