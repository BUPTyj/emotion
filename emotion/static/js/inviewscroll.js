//scroll function
$(window).on('scroll', function () {
    var trainPosition = Math.round($(window).scrollTop() / $(window).height() * 100);
    $('.scrollrow').css('transform', 'translateX(' + (trainPosition - 0) + 'px)');
 });
 
 //add remove class when item in view
 $.fn.visible = function (partial) {
    var $t = $(this),
       $w = $(window),
       viewTop = $w.scrollTop(),
       viewBottom = viewTop + $w.height(),
       SWtop = $t.offset().top,
       SWbottom = SWtop + $t.height(),
       compareTop = partial === true ? SWbottom : SWtop,
       compareBottom = partial === true ? SWtop : SWbottom;
    return ((compareBottom <= viewBottom) && (compareTop >= viewTop));
 };
 
 $(window).on('scroll', function (event) {
    $(".wide-block-scroll").each(function (i, el) {
       var el = $(el);
       if (el.visible(true)) {
          el.addClass("scrollrow");
       } else {
          el.removeClass("scrollrow");
       }
    });
 });


