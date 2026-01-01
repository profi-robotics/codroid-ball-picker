// Robot calibration plate for Codroid (units in mm).
// Marker centers follow SVG-style coordinates: origin at top-left, +x right, +y down.
plate_width_mm = 300;
plate_height_mm = 200;
plate_thickness_mm = 5;
marker_size_mm = 40;
marker_recess_mm = 1;
label_size_mm = 8;
label_depth_mm = 0.6;
label_offset_mm = 8;

marker_centers_mm = [
  [10, 100, 80],
  [11, 200, 80],
  [12, 200, 150]
];

robot_sequence = [
  [10, 1],
  [11, 2],
  [12, 3]
];

function to_scad(point_xy) = [point_xy[0], plate_height_mm - point_xy[1]];
function center_for(marker_id) = [for (marker = marker_centers_mm) if (marker[0] == marker_id) [marker[1], marker[2]]][0];

module marker_recess(center_xy) {
  let(scad_xy = to_scad(center_xy))
    translate([
      scad_xy[0] - marker_size_mm / 2,
      scad_xy[1] - marker_size_mm / 2,
      plate_thickness_mm - marker_recess_mm
    ])
      cube([marker_size_mm, marker_size_mm, marker_recess_mm + 0.1]);
}

module sequence_label(center_xy, label_text) {
  label_xy = [center_xy[0], center_xy[1] + marker_size_mm / 2 + label_offset_mm];
  scad_xy = to_scad(label_xy);
  translate([scad_xy[0], scad_xy[1], plate_thickness_mm - label_depth_mm])
    linear_extrude(height=label_depth_mm + 0.1)
      text(
        text = str(label_text),
        size = label_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "center"
      );
}

difference() {
  cube([plate_width_mm, plate_height_mm, plate_thickness_mm]);
  for (marker = marker_centers_mm) {
    marker_recess([marker[1], marker[2]]);
  }
  for (entry = robot_sequence) {
    sequence_label(center_for(entry[0]), entry[1]);
  }
}
