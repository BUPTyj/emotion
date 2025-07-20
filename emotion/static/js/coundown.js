/* Main JS file of the template*/
(function ($) {
	"use strict";
 
//timer
jQuery(document).ready(function($) {
    function makeTimer() {
        var platformTimeZone = "America/New_York";
  // EST = New_York
  // CST = Chicago
  // MST = Denver
  // PST = Los_Angeles
        var endTime = new Date("july 25, 2025 12:00:00 am" );
        endTime = (Date.parse(endTime)) / 1000;
        var now = new Date().toLocaleString('en-US', { timeZone: platformTimeZone });
        var now = (Date.parse(now) / 1000);

        var timeLeft = endTime - now;

        var days = Math.floor(timeLeft / 86400);
        var hours = Math.floor((timeLeft - (days * 86400)) / 3600);
        var minutes = Math.floor((timeLeft - (days * 86400) - (hours * 3600 )) / 60);
        var seconds = Math.floor((timeLeft - (days * 86400) - (hours * 3600) - (minutes * 60)));

        if (hours < "10") { hours = "0" + hours; }
        if (minutes < "10") { minutes = "0" + minutes; }
        if (seconds < "10") { seconds = "0" + seconds; }

        $(".ct-time-days").html(days);
        $(".ct-time-hours").html(hours);
        $(".ct-time-minutes").html(minutes);
        $(".ct-time-seconds").html(seconds);
    }
    setInterval(function() { makeTimer(); }, 1000);
});
 

})(jQuery);