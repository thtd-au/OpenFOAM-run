from pathlib import Path


def write_block_mesh_dict(
    L: float,
    w: float,
    d: float,
    nx: int,
    ny: int,
    nz: int,
    y_refinement: float,
    output_path: str | Path,
):
    """
    Generate OpenFOAM blockMeshDict for rectangular CFD domain.

    Coordinates:
        x: width  [0, w]
        y: depth  [0, d]
        z: height [0, L]

    Boundary patches:
        z=0 : inlet
        z=L : outlet
        y=0 : cathode
        y=d : anode
        x=0 and x=w : wall

    Mesh:
        Split into two blocks in y.
        Fine near y=0 and y=d, coarse near y=d/2.
    """

    if ny % 2 != 0:
        raise ValueError("ny must be even because the mesh is split into two y-blocks.")

    if y_refinement <= 0:
        raise ValueError("y_refinement must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_mid = d / 2
    ny_half = ny // 2

    text = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale 1;

vertices
(
    (0   0      0)     // 0
    ({w} 0      0)     // 1
    ({w} {y_mid} 0)    // 2
    (0   {y_mid} 0)    // 3

    (0   0      {L})   // 4
    ({w} 0      {L})   // 5
    ({w} {y_mid} {L})  // 6
    (0   {y_mid} {L})  // 7

    ({w} {d} 0)        // 8
    (0   {d} 0)        // 9
    ({w} {d} {L})      // 10
    (0   {d} {L})      // 11
);

blocks
(
    // y = 0 to y = d/2: fine at cathode, coarse toward center
    hex (0 1 2 3 4 5 6 7)    ({nx} {ny_half} {nz})    simpleGrading (1 {y_refinement} 1)

    // y = d/2 to y = d: coarse at center, fine toward anode
    hex (3 2 8 9 7 6 10 11)  ({nx} {ny_half} {nz})    simpleGrading (1 {1 / y_refinement} 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 1 2 3)
            (3 2 8 9)
        );
    }}

    outlet
    {{
        type patch;
        faces
        (
            (4 7 6 5)
            (7 11 10 6)
        );
    }}

    cathode
    {{
        type wall;
        faces
        (
            (0 4 5 1)
        );
    }}

    anode
    {{
        type wall;
        faces
        (
            (9 8 10 11)
        );
    }}

    wall
    {{
        type wall;
        faces
        (
            (0 3 7 4)
            (1 5 6 2)
            (3 9 11 7)
            (2 6 10 8)
        );
    }}
);

mergePatchPairs
(
);
"""

    output_path.write_text(text)
    print(f"Wrote blockMeshDict to: {output_path}")


if __name__ == "__main__":
    write_block_mesh_dict(
        L=0.1,
        w=0.05,
        d=0.005,
        nx=40,
        ny=40,
        nz=100,
        y_refinement=5,
        output_path="system/blockMeshDict",
    )