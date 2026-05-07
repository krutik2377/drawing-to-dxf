import sys
import ezdxf
import math

def create_dxf_part_713(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=7)

    # Part 713: L-Section Member
    # Section: L 150x15 (assuming 150x150x15 equal angle)
    # Segment length: EST 500mm for representation
    
    leg_dim = 150
    thickness = 15
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

    # Holes: M24x75 (Ø26) for 740, M20x50 (Ø22) for 739
    hole_radius_m24 = 26/2
    hole_radius_m20 = 22/2
    gauge_line = 75 # Gauge for 150mm leg
    
    # Set 1 of holes (for 740): 4x M24
    x_coords_m24 = [50, 80, 110, 140] # EST spacing 30mm
    for x in x_coords_m24:
        msp.add_circle((x, gauge_line), hole_radius_m24, dxfattribs={'layer': 'GEOMETRY'})
        
    # Set 2 of holes (for 739): 4x M20
    x_coords_m20 = [250, 300, 350, 400] # EST spacing 50mm
    for x in x_coords_m20:
        msp.add_circle((x, gauge_line), hole_radius_m20, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 713 - L-Section Member", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 10), 5, 5)
    msp.add_mtext(f"SECTION: L {leg_dim}x{leg_dim}x{thickness} (EST 150x150x15)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 20), 5, 5)
    msp.add_mtext(f"LENGTH: EST {segment_length} mm (segment)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 30), 5, 5)
    msp.add_mtext(f"4x Ø26 (M24) for 740, 4x Ø22 (M20) for 739", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_dim + 40), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_713(sys.argv[1])