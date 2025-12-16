<?php
// Minimal project config to avoid header warnings and keep OPS open for logged-in users.
require_once("../inc/util.inc");

if (function_exists('ob_get_level') && ob_get_level() === 0) {
    ob_start();
}

define("PROJECT", "boincserver");
define("COPYRIGHT_HOLDER", "BOINC Demo");

$cfg = get_config();
$master_url = parse_config($cfg, "<master_url>");
define("URL_BASE", $master_url);
define("IMAGE_PATH", "../user_profile/images/");
define("IMAGE_URL", "user_profile/images/");
define("PROFILE_PATH", "../user_profile/");
define("PROFILE_URL", "user_profile/");
define("LANGUAGE_FILE", "languages.txt");

define("FORUM_QA_MERGED_MODE", true);
define("DISABLE_PROFILES", true);
define("USE_STOPFORUMSPAM", false);
define("TERMSOFUSE_FILE", "../../terms_of_use.txt");

function project_banner($title, $prefix, $is_main) {
    sample_navbar(secure_url_base(), get_logged_in_user(false), false);
    if ($title) {
        echo "<h2>$title</h2>\n";
    }
}

function project_footer($show_return, $show_date, $prefix) {
    echo '<br><p class="text-center"> &copy;'.gmdate("Y ").COPYRIGHT_HOLDER.'</p>';
    if ($show_date) {
        $t = time_str(time());
        echo "<center><small>".tra("Generated")." $t</small><center>\n";
    }
}

// OPS auth: require any logged-in project user.
function auth_ops() {
    $user = get_logged_in_user_ops();
    if (!$user) {
        admin_page_head("Log in");
        print_login_form_ops();
        admin_page_tail();
        exit;
    }
}

