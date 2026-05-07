import sys
import ezdxf
import math

def create_dxf_part_731(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=7)

    # Part 731: Gusset Plate (from Schnitt A-A)
    # Geometry shared with 707, 727, 728
    # Overall dimensions: 128x128 mm
    # Cut corner: 72mm from corner along both edges
    
    plate_w = 128
    plate_h = 128
    cut_dim = 72
    
    # Outline (LWPOLYLINE)
    points = [
        (0, cut_dim),
        (cut_dim, 0),
        (plate_w, 0),
        (plate_w, plate_h),
        (0, plate_h),
        (0, cut_dim)
    ]
    msp.add_lwpolyline(points, dxfattribs={'layer': 'GEOMETRY'})

    # Holes: 2x M24 (Ø26)
    hole_radius_m24 = 26/2
    holes_coords = [
        (cut_dim, plate_h - cut_dim), # (72, 56)
        (plate_w - cut_dim, cut_dim)  # (56, 72)
    ]
    for x, y in holes_coords:
        msp.add_circle((x, y), hole_radius_m24, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 731 - Gusset Plate", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 10), 5, 5)
    msp.add_mtext("MATERIAL: BLE 15x236 - 731 (Thickness 15mm)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 20), 5, 5)
    msp.add_mtext(f"OVERALL: W {plate_w} x H {plate_h} mm", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 30), 5, 5)
    msp.add_mtext(f"2x Ø26 (M24)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 40), 5, 5)
    msp.add_mtext("NOTE: Geometry shared with PART 707 (PL 6x49, Thk 6mm), PART 727 (BLE 20x28, Thk 20mm), PART 728 (BLE 15x236, Thk 15mm)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 50), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_731(sys.argv[1])