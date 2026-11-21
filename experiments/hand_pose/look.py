"""A red concert-ukulele look, after the reference photos.

Runs the project's decorate() (strum/presentation.py) and then restyles what it
added: glossy red wood with grain, a near-black fretboard with white dots,
bright frets, white nut and saddle, a dark bridge, an ornate rosette, a red
headstock with black tuner buttons and silver posts, white strings, and a light
grey studio backdrop with brighter lighting. Visual only: every shape changed
here has collisions off, and the instrument's collision geometry is untouched.

install() swaps it in for strum.presentation.decorate, in memory only.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strum.presentation as presentation  # noqa: E402

ORIGINAL = presentation.decorate

RED = '.40 .055 .05 1'         # gloss wine-red mahogany (neck, headstock)
RED_DARK = '.28 .035 .03 1'    # sides, binding
EBONY = '.035 .03 .03 1'       # fretboard, bridge
IVORY = '.96 .95 .91 1'        # nut, saddle, inlay dots
SILVER = '.86 .87 .89 1'       # frets, tuner posts and washers
BLACK = '.03 .03 .03 1'        # tuner buttons
STRING = '.97 .97 .95 1'


def restyle(root):
    uke = root.find(".//body[@name='ukulele']")
    asset = root.find('asset')
    geom = lambda name: uke.find(f"geom[@name='{name}']")

    # Wood: the existing grain texture, tinted red and made glossy.
    soundboard = asset.find("material[@name='koa_top']")
    soundboard.set('rgba', '.52 .075 .07 1'); soundboard.set('specular', '.45'); soundboard.set('shininess', '.7')
    ET.SubElement(asset, 'material', name='red_gloss', rgba=RED, specular='.5', shininess='.75')
    ET.SubElement(asset, 'material', name='red_sides', texture='koa_grain', rgba='.46 .07 .06 1',
                  specular='.4', shininess='.65')
    ET.SubElement(asset, 'material', name='ebony', rgba=EBONY, specular='.3', shininess='.5')
    ET.SubElement(asset, 'material', name='metal', rgba=SILVER, specular='1', shininess='1', reflectance='.2')
    geom('body_sides').attrib.pop('rgba', None); geom('body_sides').set('material', 'red_sides')
    for g in uke.findall('geom'):
        name = g.get('name', '')
        if name.startswith('binding_'):
            g.set('rgba', RED_DARK)
    for name, color in [('rosette_outer', '.22 .05 .04 1'), ('rosette_light', '.78 .58 .38 1'),
                        ('rosette_inner', '.22 .05 .04 1'), ('soundhole', '.02 .012 .01 1')]:
        geom(name).set('rgba', color)
    # A second, finer light ring, for a mosaic-like rosette.
    outer = geom('rosette_outer')
    x = outer.get('pos').split()[0]
    ET.SubElement(uke, 'geom', name='rosette_fine', type='cylinder', contype='0', conaffinity='0', group='2',
                  mass='0', pos=f'{x} 0 .0004', size='.0215 .00012', rgba='.90 .80 .62 1')
    geom('rosette_inner').set('pos', f'{x} 0 .0005')
    geom('soundhole').set('pos', f'{x} 0 .0007')

    # Neck, fretboard, headstock, hardware.
    geom('fretboard').attrib.pop('rgba', None); geom('fretboard').set('material', 'ebony')
    geom('neck_wood').attrib.pop('rgba', None); geom('neck_wood').set('material', 'red_gloss')
    uke.remove(geom('headstock'))                   # replaced by a shaped headstock in details()
    for name in ('nut_visual', 'saddle'):
        geom(name).set('rgba', IVORY)
    geom('bridge').attrib.pop('rgba', None); geom('bridge').set('material', 'ebony')
    for g in uke.findall('geom'):
        name = g.get('name', '')
        if name.startswith(('tuner_washer_', 'tuner_post_')):
            g.attrib.pop('rgba', None); g.set('material', 'metal')
        elif name.startswith('tuner_button_'):
            g.set('rgba', BLACK)
        elif name.startswith('inlay_'):
            g.set('rgba', IVORY)
        elif name.startswith('fret_visual_'):
            g.attrib.pop('rgba', None); g.set('material', 'metal')
        elif name.startswith('string_'):
            g.set('rgba', STRING)

    # Light grey studio backdrop, brighter and warmer lights.
    sky = asset.find("texture[@name='presentation_sky']")
    sky.set('rgb1', '.86 .86 .84'); sky.set('rgb2', '.62 .62 .60')
    world = root.find('worldbody')
    world.find("geom[@name='ground']").set('rgba', '.78 .78 .76 1')
    for light in list(world.findall('light')):
        world.remove(light)
    for pos, direction, diffuse in [('.2 -1.2 2.2', '0 .5 -1', '.62 .60 .56'),
                                    ('-.8 -.6 1.6', '.4 .3 -1', '.30 .31 .34'),
                                    ('.6 .8 2', '-.2 -.3 -1', '.22 .22 .24')]:
        ET.SubElement(world, 'light', pos=pos, dir=direction, diffuse=diffuse, ambient='.16 .16 .16',
                      castshadow='false')


# Concert body outline, measured from the reference photo using the nut-to-bridge distance (381 mm)
# as the ruler. Half-widths in mm along x (mm from the nut). Cubic Bezier pieces, smooth joins.
OUTLINE_MM = [
    [(185, 0), (185, 42), (196, 70), (217, 70)],      # shoulder up to the upper bout (140 mm wide)
    [(217, 70), (240, 70), (252, 57), (272, 57)],     # into the waist (114 mm wide)
    [(272, 57), (300, 57), (328, 106), (387, 106)],   # out to the lower bout (212 mm wide), at the bridge
    [(387, 106), (440, 106), (460, 64), (460, 0)],    # round end, 80 mm past the bridge
]
DEPTH = .065                 # body depth, m
HOLE_X, HOLE_R = .291, .025  # sound hole centre and radius, m
BOARD_END = .245             # fretboard runs onto the body up to here, m


def extrude(asset, name, outline, bottom, top, center, uv):
    """Flat-capped prism mesh from a clockwise outline (seen from +z).

    The caps and the side wall use separate copies of the outline points, so
    shading does not blend between the flat top and the sides at the edge.
    Every face points outward.
    """
    n = len(outline)
    ring = lambda z: [(x, y, z) for x, y in outline]
    vertices = ring(bottom) + ring(top) + [(*center, bottom), (*center, top)]   # caps
    side = len(vertices)
    vertices += ring(bottom) + ring(top)                                         # side wall
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(2 * n, i, j), (2 * n + 1, n + j, n + i)]
        faces += [(side + i, side + n + j, side + j), (side + i, side + n + i, side + n + j)]
    coords = [uv(v[0], v[1]) for v in vertices]
    ET.SubElement(asset, 'mesh', name=name, vertex=' '.join(str(v) for xyz in vertices for v in xyz),
                  face=' '.join(str(v) for f in faces for v in f),
                  texcoord=' '.join(str(v) for xy in coords for v in xy))
    return name


def bezier(points, n=20):
    q = np.asarray(points, dtype=float)
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    return (1 - t) ** 3 * q[0] + 3 * (1 - t) ** 2 * t * q[1] + 3 * (1 - t) * t * t * q[2] + t ** 3 * q[3]


def visual(parent, name, kind, **attrs):
    return ET.SubElement(parent, 'geom', name=name, type=kind, contype='0', conaffinity='0', group='2',
                         mass='0', **attrs)


def reshape_body(root, p):
    uke = root.find(".//body[@name='ukulele']")
    asset = root.find('asset')
    scale = p.scale_length / .381
    half = np.vstack([bezier(piece) for piece in OUTLINE_MM] + [np.array([[460., 0.]])]) / 1000 * scale
    outline = np.vstack([half, half[-2:0:-1] * [1, -1]])
    x0, x1, w = outline[:, 0].min(), outline[:, 0].max(), outline[:, 1].max()

    def prism(name, bottom, top):
        # The waist centre sees the whole outline, so a fan from it is valid.
        return extrude(asset, name, outline, bottom, top, (.28 * scale, 0.),
                       lambda x, y: ((x - x0) / (x1 - x0), (y + w) / (2 * w)))

    uke.find("geom[@name='body_sides']").set('mesh', prism('concert_body', -DEPTH, -.0015))
    uke.find("geom[@name='soundboard']").set('mesh', prism('concert_top', -.0015, -.0006))
    for g in list(uke.findall('geom')):
        if g.get('name', '').startswith('binding_'):
            uke.remove(g)
    for i, (a, b) in enumerate(zip(outline, np.roll(outline, -1, axis=0))):
        visual(uke, f'edge_{i}', 'capsule', fromto=f'{a[0]} {a[1]} -.0008 {b[0]} {b[1]} -.0008',
               size='.0012', material='red_sides')

    # Sound hole and a wide mosaic rosette.
    hx = HOLE_X * scale
    for g in list(uke.findall('geom')):
        if g.get('name', '').startswith(('rosette_', 'soundhole')):
            uke.remove(g)
    visual(uke, 'rosette_band', 'cylinder', pos=f'{hx} 0 .0001', size='.0365 .00012', rgba='.16 .035 .03 1')
    for k in range(48):                                   # alternating tiles around the band
        a = 2 * np.pi * k / 48
        r = .0315
        visual(uke, f'rosette_tile_{k}', 'box', pos=f'{hx + r * np.cos(a)} {r * np.sin(a)} .0003',
               euler=f'0 0 {a}', size='.0035 .0018 .00012',
               rgba='.62 .40 .24 1' if k % 2 else '.30 .07 .05 1')
    visual(uke, 'rosette_ring_out', 'cylinder', pos=f'{hx} 0 .0002', size='.0362 .00012', rgba='.80 .64 .42 1')
    visual(uke, 'rosette_ring_out_dark', 'cylinder', pos=f'{hx} 0 .00025', size='.0352 .00012',
           rgba='.16 .035 .03 1')
    visual(uke, 'rosette_ring_in', 'cylinder', pos=f'{hx} 0 .0004', size='.0278 .00012', rgba='.80 .64 .42 1')
    visual(uke, 'rosette_ring_in_dark', 'cylinder', pos=f'{hx} 0 .00045', size='.0268 .00012',
           rgba='.16 .035 .03 1')
    visual(uke, 'soundhole', 'cylinder', pos=f'{hx} 0 .0006', size=f'{HOLE_R * scale} .00012',
           rgba='.07 .035 .02 1')

    # Fretboard runs onto the body, with frets and the usual dots (5, 7, 10, 12 double, 15).
    end = BOARD_END * scale
    visual(uke, 'fretboard_extension', 'box', pos=f'{(p.neck_length + end) / 2} 0 -.002',
           size=f'{(end - p.neck_length) / 2} {p.neck_width / 2} .002', material='ebony')
    fret = lambda n: p.scale_length * (1 - 2 ** (-n / 12))
    for n in range(p.n_frets + 1, 20):
        if fret(n) >= end - .002:
            break
        visual(uke, f'fret_extra_{n}', 'capsule', material='metal', size=str(p.fret_radius),
               fromto=f'{fret(n)} {-p.neck_width / 2} {p.fret_top - p.fret_radius} '
                      f'{fret(n)} {p.neck_width / 2} {p.fret_top - p.fret_radius}')
    for g in list(uke.findall('geom')):
        if g.get('name', '').startswith('inlay_'):
            uke.remove(g)
    for n, ys in [(5, [0]), (7, [0]), (10, [0]), (12, [-.008, .008]), (15, [0])]:
        x = (fret(n - 1) + fret(n)) / 2
        for k, y in enumerate(ys):
            visual(uke, f'dot_{n}_{k}', 'cylinder', pos=f'{x} {y} .00015', size='.0019 .00015', rgba=IVORY)

    # Wide dark bridge with ivory saddle and four white string ties.
    for g in list(uke.findall('geom')):
        if g.get('name', '') == 'bridge':
            uke.remove(g)
    L = p.scale_length
    visual(uke, 'bridge', 'box', pos=f'{L + .005} 0 .0012', size=f'.008 {.040 * scale} .0014', material='ebony')
    from strum.ukulele import string_y
    for i in range(4):
        visual(uke, f'string_tie_{i}', 'cylinder', pos=f'{L + .0095} {string_y(p, i, L)} .0027',
               size='.0016 .0002', rgba=IVORY)


TEXTURE = ROOT / 'outputs/hand-pose/textures/mahogany.png'


def write_png(path, rgb):
    import struct
    import zlib
    height, width, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def make_grain(path=TEXTURE, width=512, height=1024, seed=7):
    """Wine-red mahogany with straight grain running along the image's x (the body's length)."""
    rng = np.random.default_rng(seed)
    y = np.arange(height)[:, None].astype(float)
    x = np.arange(width)[None, :].astype(float)
    # Grain lines: irregular spacing across y, a gentle wave along x.
    wave = 3 * np.sin(2 * np.pi * x / width * 1.3 + 1.1) + 1.5 * np.sin(2 * np.pi * x / width * 3.7)
    phase = np.cumsum(rng.uniform(.45, 1.05, height))[:, None]          # varying line spacing
    lines = np.sin(phase + wave * .35)
    fine = rng.normal(0, 1, (height, width))
    fine = (fine + np.roll(fine, 1, 1) + np.roll(fine, 2, 1) + np.roll(fine, 3, 1)) / 4   # streaks along x
    bands = np.sin(2 * np.pi * y / height * 5 + rng.uniform(0, 6)) * .5 + np.sin(2 * np.pi * y / height * 13) * .3
    shade = 1 + .10 * lines ** 3 + .05 * fine + .06 * bands
    base = np.array([118, 18, 16], dtype=float)                         # wine-red mahogany
    rgb = np.clip(base[None, None, :] * shade[..., None], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, np.ascontiguousarray(rgb))
    return path


def details(root, p):
    """Grain, rounded glossy edges, neck heel and a shaped headstock."""
    uke = root.find(".//body[@name='ukulele']")
    asset = root.find('asset')
    texture = TEXTURE if TEXTURE.exists() else make_grain()
    ET.SubElement(asset, 'texture', name='mahogany', type='2d', file=str(texture))
    ET.SubElement(asset, 'material', name='mahogany_top', texture='mahogany', rgba='1 1 1 1',
                  specular='.5', shininess='.85', reflectance='.04')
    ET.SubElement(asset, 'material', name='mahogany_side', texture='mahogany', rgba='.82 .82 .82 1',
                  specular='.55', shininess='.85')
    ET.SubElement(asset, 'material', name='mahogany_edge', rgba='.44 .06 .055 1', specular='.9', shininess='.95')
    uke.find("geom[@name='soundboard']").set('material', 'mahogany_top')
    uke.find("geom[@name='body_sides']").set('material', 'mahogany_side')

    # Rounded, glossy edges: thick capsules along the top and back rims, replacing the thin edge line.
    edges = [g for g in uke.findall('geom') if g.get('name', '').startswith('edge_')]
    rim = []
    for g in edges:
        a = np.array(g.get('fromto').split(), dtype=float)
        rim.append(a[:2])
        uke.remove(g)
    for i, (a, b) in enumerate(zip(rim, rim[1:] + rim[:1])):
        for label, z in (('top', -.0034), ('back', -DEPTH + .003)):
            visual(uke, f'rim_{label}_{i}', 'capsule', material='mahogany_edge', size='.003',
                   fromto=f'{a[0]} {a[1]} {z} {b[0]} {b[1]} {z}')

    # Neck heel: where the neck meets the body at the back.
    scale = p.scale_length / .381
    visual(uke, 'neck_heel', 'ellipsoid', material='red_gloss', pos=f'{.188 * scale} 0 {-.032}',
           size=f'.016 {p.neck_width / 2 - .002} .030')

    # Shaped headstock: flares slightly toward a gently curved crown with rounded corners.
    # Walked clockwise seen from the fretboard side (x toward the body, y toward the A string):
    # down the nut end, along the A side, round corner, curved crown, round corner, back along the G side.
    r, cx = .004, -.058
    pts = [(-.002, .019), (-.002, -.019), (cx, -.0215)]
    pts += [(cx + r * np.cos(a), -.0175 + r * np.sin(a)) for a in np.linspace(-np.pi / 2, -np.pi, 6)[1:]]
    pts += [(-.062 - .006 * (1 - t * t), .0175 * t) for t in np.linspace(-1, 1, 11)[1:-1]]
    pts += [(cx + r * np.cos(a), .0175 + r * np.sin(a)) for a in np.linspace(np.pi, np.pi / 2, 6)[:-1]]
    pts += [(cx, .0215)]
    pts = np.array(pts)
    extrude(asset, 'concert_headstock', pts, -.014, -.002, tuple(pts.mean(axis=0)),
            lambda x, y: ((x + .07) / .07, (y + .025) / .05))
    visual(uke, 'headstock_shaped', 'mesh', mesh='concert_headstock', material='mahogany_top')


def decorate_concert(root, p, repo):
    ORIGINAL(root, p, repo)
    restyle(root)
    reshape_body(root, p)
    details(root, p)


def install():
    presentation.decorate = decorate_concert
