import sys
import ezdxf
import math

def create_dxf_part_721(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=7)

    # Part 721: L-Section Member
    # Section: L 250x28 (assuming 250x250x28 equal angle)
    # Segment length: EST 500mm for representation
    
    leg_dim = 250
    thickness = 28
    segment_length = 500 # EST length for drawing
    
    # Outline of L-section (equal angle)
    points = [
        (0, 0),
        (segment_length, 0),
        (segment_length, thickness),
        (thickness, thickness),
        (thickness, leg_dim),
        (0, leg_dim),
        (0, 0)
    ]
    msp.add_lwpolyline(points, dxfattribs={'layer': 'GEOMETRY'})

    # Holes: M27x80 (Ø29)
    # 4 holes in a pattern on the 250mm leg
    # Gauge line: 75mm from back of angle (EST, typical for 250mm leg)
    # Spacing: 60mm, 100mm gap, 60mm (from drawing interpretation)
    hole_radius_m27 = 29/2
    gauge_line = 75
    
    holes_x_coords = [50, 50+60, 50+60+100, 50+60+100+60] # 50, 110, 210, 270
    
    for x in holes_x_coords:
        msp.add_circle((x, gauge_line), hole_radius_m27, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 721 - L-Section Member", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 10), 5, 5)
    msp.add_mtext(f"SECTION: L {leg_dim}x{leg_dim}x{thickness} (EST 250x250x28)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 20), 5, 5)
    msp.add_mtext(f"LENGTH: EST {segment_length} mm (segment)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 30), 5, 5)
    msp.add_mtext(f"4x Ø29 (M27)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 40), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_721(sys.argv[1])