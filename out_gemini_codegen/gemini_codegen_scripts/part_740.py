import sys
import ezdxf
import math

def create_dxf_part_740(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=7)

    # Part 740: Connection Plate (from main elevation)
    # Material: BLE 20x150 - 740
    # Overall dimensions: Width 150mm, Height EST 150mm
    
    plate_w = 150
    plate_h = 150 # EST height based on typical connection plate proportions
    thickness = 20
    
    # Outline (LWPOLYLINE)
    points = [
        (0, 0),
        (plate_w, 0),
        (plate_w, plate_h),
        (0, plate_h),
        (0, 0)
    ]
    msp.add_lwpolyline(points, dxfattribs={'layer': 'GEOMETRY'})

    # Holes: 8x M24 (Ø26)
    # 4 holes for L 150x15 (horizontal) - top part
    # 4 holes for L 150x15 (diagonal) - bottom part
    hole_radius_m24 = 26/2
    gauge_line_h = 75 # Gauge for 150mm leg
    
    # Holes for horizontal member (top part of plate)
    # X-coords: EST 30, 60, 90, 120 (spacing 30mm)
    x_coords_h = [30, 60, 90, 120]
    for x in x_coords_h:
        msp.add_circle((x, plate_h - gauge_line_h), hole_radius_m24, dxfattribs={'layer': 'GEOMETRY'})
        
    # Holes for diagonal member (bottom part of plate)
    # X-coords: EST 30, 60, 90, 120 (spacing 30mm)
    x_coords_d = [30, 60, 90, 120]
    for x in x_coords_d:
        msp.add_circle((x, gauge_line_h), hole_radius_m24, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 740 - Connection Plate", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 10), 5, 5)
    msp.add_mtext(f"MATERIAL: BLE {thickness}x{plate_w} - 740", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 20), 5, 5)
    msp.add_mtext(f"OVERALL: W {plate_w} x H {plate_h} mm (H EST)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 30), 5, 5)
    msp.add_mtext(f"8x Ø26 (M24)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 40), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_740(sys.argv[1])