import cv2
import numpy as np


def debug_print(debug, *args):
    if debug:
        print(*args)

def debug_imshow(debug, window_name, image):
    if debug:
        cv2.imshow(window_name, image)

def opencv_method(path, debug=False):
    debug_print(debug, f"\n=== Kép betöltése: {path} ===")

    # Kép betöltése - Szürke árnyalatos
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"A kép nem található: {path}")

    output = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    # Zajszűrés TODO: Kisebb/nagyobb, ha nagyon zajos a kép akkor rossz
    blur = cv2.GaussianBlur(img, (7, 7), 0)

    # Küszöbölés
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5
    )

    # Morfológiai tisztítás # TODO: Más kombináció, ha a zajszűrés megfelelő mennyiségű, de az eredmény továbbra is rossz
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # Kontúrkeresés
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    debug_print(
        debug,
        f"\nTalált külső kontúrok száma: {len(contours)}"
    )

    dice_results = []

    contour_debug = cv2.cvtColor(
        binary.copy(),
        cv2.COLOR_GRAY2BGR
    )

    for contour_index, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        debug_print(debug, "\n--------------------------------")
        debug_print(debug, f"Kontúr #{contour_index}")
        debug_print(debug, f"Terület: {area:.1f}")

        if area < 1000:
            debug_print(
                debug,
                "Elutasítva: túl kis terület"
            )
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / h

        debug_print(
            debug,
            f"x={x}, y={y}, w={w}, h={h}"
        )

        debug_print(
            debug,
            f"Aspect ratio: {aspect_ratio:.2f}"
        )

        if not (0.7 < aspect_ratio < 1.3):

            debug_print(
                debug,
                "Elutasítva: nem négyzet alakú"
            )

            continue

        debug_print(
            debug,
            "Kocka jelölt elfogadva"
        )

        cv2.drawContours(
            contour_debug,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            contour_debug,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        roi = binary[y:y + h, x:x + w]

        blob_contours, _ = cv2.findContours(
            roi,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Az eredeti képből kivágott kocka
        roi_original = img[y:y + h, x:x + w]

        # Erről készül a debug kép
        blob_debug = cv2.cvtColor(
            roi_original.copy(),
            cv2.COLOR_GRAY2BGR
        )

        # Felnagyítás a debughoz
        scale = 4

        blob_debug_large = cv2.resize(
            blob_debug,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        dot_count = 0

        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)

        debug_print(
            debug,
            f"Talált blobok száma: {len(blob_contours)}"
        )

        for blob_index, blob in enumerate(blob_contours):

            blob_area = cv2.contourArea(blob)

            bx, by, bw, bh = cv2.boundingRect(blob)

            debug_print(debug, f"\nBlob {blob_index}")

            debug_print(
                debug,
                f"area={blob_area:.1f}, "
                f"x={bx}, y={by}, "
                f"w={bw}, h={bh}"
            )

            M = cv2.moments(blob)

            if M["m00"] == 0:

                debug_print(
                    debug,
                    "Elutasítva: nulla momentum"
                )

                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            debug_print(
                debug,
                f"Középpont: ({cx}, {cy})"
            )

            perimeter = cv2.arcLength(
                blob,
                True
            )

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * blob_area
                / (perimeter * perimeter)
            )

            valid_area = (
                50 < blob_area < 2000
                and circularity > 0.70
            )

            inside_margin = (
                margin_x < cx < w - margin_x
                and
                margin_y < cy < h - margin_y
            )

            debug_print(
                debug,
                f"Circularity: {circularity:.2f}"
            )

            debug_print(
                debug,
                f"Méret OK: {valid_area}"
            )

            debug_print(
                debug,
                f"Marginon belül: {inside_margin}"
            )

            if valid_area and inside_margin:

                dot_count += 1

                cv2.rectangle(
                    blob_debug_large,
                    (bx * scale, by * scale),
                    ((bx + bw) * scale, (by + bh) * scale),
                    (0, 255, 0),
                    3
                )

                cv2.circle(
                    blob_debug_large,
                    (cx * scale, cy * scale),
                    8,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    blob_debug_large,
                    f"DOT {dot_count}",
                    (bx * scale, max(20, by * scale - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    blob_debug_large,
                    f"A:{blob_area:.0f}",
                    (bx * scale, by * scale + bh * scale + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                cv2.putText(
                    blob_debug_large,
                    f"C:{circularity:.2f}",
                    (bx * scale, by * scale + bh * scale + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                debug_print(
                    debug,
                    "-> ELFOGADVA"
                )

            else:

                cv2.rectangle(
                    blob_debug_large,
                    (bx * scale, by * scale),
                    ((bx + bw) * scale, (by + bh) * scale),
                    (0, 0, 255),
                    3
                )

                cv2.circle(
                    blob_debug_large,
                    (cx * scale, cy * scale),
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    blob_debug_large,
                    "REJECT",
                    (bx * scale, max(20, by * scale - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    blob_debug_large,
                    f"A:{blob_area:.0f}",
                    (bx * scale, by * scale + bh * scale + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                cv2.putText(
                    blob_debug_large,
                    f"C:{circularity:.2f}",
                    (bx * scale, by * scale + bh * scale + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                debug_print(
                    debug,
                    "-> ELUTASÍTVA"
                )

        debug_print(
            debug,
            f"\nVégső pontszám: {dot_count}"
        )

        dice_results.append(
            (x, y, w, h, dot_count)
        )

        debug_imshow(
            debug,
            f"Dice_Debug_{contour_index}",
            blob_debug_large
        )

    # Balról jobbra rendezés
    dice_results = sorted(
        dice_results,
        key=lambda d: d[0]
    )

    total = 0

    for idx, (x, y, w, h, value) in enumerate(dice_results):

        total += value

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Dice {idx + 1}: {value}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    debug_print(debug, "\n===================================")
    debug_print(debug, "Felismerések összesítése")
    debug_print(debug, "===================================")

    for i, (_, _, _, _, value) in enumerate(dice_results):

        debug_print(
            debug,
            f"Kocka {i + 1}: {value}"
        )

    debug_print(
        debug,
        f"\nÖsszeg: {total}"
    )

    cv2.putText(
        output,
        f"Total: {total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    debug_imshow(debug, "Original", img)
    debug_imshow(debug, "Blur", blur)
    debug_imshow(debug, "Threshold", thresh)
    debug_imshow(debug, "Binary", binary)
    debug_imshow(debug, "Dice Contours", contour_debug)
    debug_imshow(debug, "Result", output)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output

# TODO : Az összes képre lefuttatni automatikusan és kiértékelni (Helyes/Összes és Helytelen/Összes arány)