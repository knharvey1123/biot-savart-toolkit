import numpy as np


class ArcSplineInterpolator:
    @staticmethod
    def circular_arc_through_points(p1, p2, p3, num_points=30):
        """
        Create circular arc through three 3D points
        Returns array of points along the arc
        """
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

        # Vectors from p1 to other points
        v1 = p2 - p1
        v2 = p3 - p1

        # Check for collinearity
        cross_product = np.cross(v1, v2)
        if np.linalg.norm(cross_product) < 1e-10:
            # Points are collinear, return straight line
            t_vals = np.linspace(0, 1, num_points)
            return np.array([p1 + t * (p3 - p1) for t in t_vals])

        # Normal to the plane containing the three points
        normal = cross_product / np.linalg.norm(cross_product)

        # Find circumcenter using perpendicular bisector method
        mid1 = (p1 + p2) / 2
        mid2 = (p2 + p3) / 2

        # Directions perpendicular to the sides in the plane
        dir1 = np.cross(v1, normal)
        dir1 = dir1 / np.linalg.norm(dir1)

        v3 = p3 - p2
        dir2 = np.cross(v3, normal)
        dir2 = dir2 / np.linalg.norm(dir2)

        # Solve for intersection of perpendicular bisectors
        # mid1 + t1 * dir1 = mid2 + t2 * dir2
        # This gives us a 3x2 system, we solve using least squares
        A = np.column_stack([dir1, -dir2])
        b = mid2 - mid1

        try:
            # Use least squares to solve the overdetermined system
            params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            t1 = params[0]
            center = mid1 + t1 * dir1
        except np.linalg.LinAlgError:
            # Fallback to linear interpolation
            t_vals = np.linspace(0, 1, num_points)
            return np.array([p1 + t * (p3 - p1) for t in t_vals])

        radius = np.linalg.norm(p1 - center)

        # Create orthonormal basis for the circle
        u = (p1 - center) / radius
        v = np.cross(normal, u)

        # Find angles for the three points
        def get_angle(point):
            rel_pos = point - center
            return np.arctan2(np.dot(rel_pos, v), np.dot(rel_pos, u))

        angle1 = get_angle(p1)
        angle2 = get_angle(p2)
        angle3 = get_angle(p3)

        # Determine the correct arc direction
        # We want the arc that goes from p1 through p2 to p3
        start_angle = angle1
        end_angle = angle3

        # Check if we need to adjust for the shorter arc
        if abs(end_angle - start_angle) > np.pi:
            if end_angle > start_angle:
                end_angle -= 2 * np.pi
            else:
                end_angle += 2 * np.pi

        # Ensure we pass through p2
        mid_angle = angle2
        if not (min(start_angle, end_angle) <= mid_angle <= max(start_angle, end_angle)):
            # Need to go the other way
            if end_angle > start_angle:
                end_angle -= 2 * np.pi
            else:
                end_angle += 2 * np.pi

        # Generate points along the arc
        t_vals = np.linspace(0, 1, num_points)
        angles = start_angle + t_vals * (end_angle - start_angle)

        arc_points = []
        for angle in angles:
            point = center + radius * (u * np.cos(angle) + v * np.sin(angle))
            arc_points.append(point)

        return np.array(arc_points), center, radius

    @staticmethod
    def get_arc_parameters(p1, p2, p3):
        """
        Extract arc parameters: radius, L1 (p1 to p2 arc length), L2 (p2 to p3 arc length)
        """
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

        # Get arc points and parameters
        try:
            arc_points, center, radius = ArcSplineInterpolator.circular_arc_through_points(p1, p2, p3, num_points=100)

            # Calculate angles for arc length computation
            def get_angle(point):
                rel_pos = point - center
                u = (p1 - center) / radius
                normal = np.cross(p2 - p1, p3 - p1)
                if np.linalg.norm(normal) > 1e-10:
                    normal = normal / np.linalg.norm(normal)
                    v = np.cross(normal, u)
                    return np.arctan2(np.dot(rel_pos, v), np.dot(rel_pos, u))
                else:
                    # Fallback for collinear points
                    return 0

            angle1 = get_angle(p1)
            angle2 = get_angle(p2)
            angle3 = get_angle(p3)

            # Calculate arc lengths
            def angle_distance(a1, a2):
                diff = a2 - a1
                return abs(diff) if abs(diff) <= np.pi else 2*np.pi - abs(diff)

            L1 = radius * angle_distance(angle1, angle2)  # Arc length from p1 to p2
            L2 = radius * angle_distance(angle2, angle3)  # Arc length from p2 to p3

            return radius, L1, L2

        except np.linalg.LinAlgError:
            # Fallback to straight line distances
            L1 = np.linalg.norm(p2 - p1)
            L2 = np.linalg.norm(p3 - p2)
            # Use average as radius approximation
            radius = (L1 + L2) / (2 * np.pi) if (L1 + L2) > 0 else 0.01
            return radius, L1, L2
