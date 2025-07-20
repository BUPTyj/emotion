/* -------------------------------------------------------------------
 * Template         : Blueket - Digital Agency & Portfolio HTML Template
 * Author           : SeparateWeb
 * File             : custom.js
------------------------------------------------------------------- */
/*1. Sticky Header    2. Slider Home services     3. Slider Home portfolio        4. Slider Tech Icon      5. Review Slider     6. logo slider     7. Work slider
8. MagnificPopup      9. Wow Animation      10. Portfolio Filter      11. Odometer      12. Background Inline Image*/

(function ($) {
  "use strict";

//Sticky Header 
function updateScroll() {
  if ($(window).scrollTop() >= 80) {
    $(".header").addClass('sticky');
  } else {
    $(".header").removeClass("sticky");
  }
}
$(function () {
  $(window).scroll(updateScroll);
  updateScroll();
});

//demo 7 hero
$(window).on("load", function () {
  $('.service--cards').owlCarousel({
      items: 4,
      nav: false,
      dots: false,
      autoplay: true,
      loop: true,   
      center: false,   
      margin: 30,
      autoplayTimeout: 3000,
      autoplayHoverPause: true,
      autoHeight: true,
      smartSpeed: 2000,   
     responsive: {
      0: {
        items: 1
      },
      520: {
        items: 2
      },
      768: {
        items: 2
      },
      900: {
        items: 3
      },
      1300: {
        items: 3
      },
      1400: {
        items: 3
      },
      1600: {
        items: 3
      },
    }
  });
});

//Home services
$(window).on("load", function () {
    $('.service-cards').owlCarousel({
        items: 6,
        nav: false,
        dots: false,
        autoplay: true,
        loop: true,   
        center: false,   
        margin: 30,
        autoplayTimeout: 3000,
        autoplayHoverPause: true,
        autoHeight: true,
        smartSpeed: 2000,   
       responsive: {
        0: {
          items: 1
        },
        520: {
          items: 2
        },
        768: {
          items: 3
        },
        900: {
          items: 4
        },
        1300: {
          items: 5
        },
        1400: {
          items: 5
        },
        1600: {
          items: 6
        },
      }
    });
});

//Home portfolio
$(window).on("load", function () {
    $('.work-slide').owlCarousel({        
        nav: true,
        navText : ["<i class='fa-solid fa-arrow-left-long'></i>","<i class='fa-solid fa-arrow-right-long'></i>"],
        dots: false,
        autoplay: true,
        loop: true,   
        center: false,   
        margin: 30,
        autoplayTimeout: 3000,
        autoplayHoverPause: true,
        autoHeight: true,
        smartSpeed: 2000,   
       responsive: {
        0: {
          items: 1
        },
        520: {
          items: 2
        },
        768: {
          items: 3
        },
        1200: {
          items: 4
        },
        1400: {
          items: 4
        },
        1600: {
          items: 4
        },
      }
    });
 });
  
//Tech Icon
$(window).on("load", function () {
    $('.sw-icon-slider').owlCarousel({       
        nav: false,
        dots: false,
        autoplay: true,
        loop: true,   
        center: false,   
        margin: 20,
        autoplayTimeout: 3000,
        autoplayHoverPause: false,
        autoHeight: true,
        smartSpeed: 2000,   
       responsive: {
        0: {
          items: 3
        },
        520: {
          items: 3
        },
        768: {
          items: 4
        },
        1200: {
          items: 6
        },
        1400: {
          items: 7
        },
        1600: {
          items: 8
        },
      }
    });
  });

//Review Slider
$(window).on("load", function () {
  $('.review-slider').owlCarousel({       
      nav: true,
      navText : ["<i class='fa-solid fa-arrow-left-long'></i>","<i class='fa-solid fa-arrow-right-long'></i>"],
      dots: false,
      autoplay: true,
      loop: true,   
      center: false,   
      margin: 30,
      autoplayTimeout: 3500,
      autoplayHoverPause: true,
      autoHeight: true,
      smartSpeed: 2000,   
     responsive: {
      0: {
        items: 1
      },
      520: {
        items: 1
      },
      768: {
        items: 1
      },
      1200: {
        items: 2
      },
      1400: {
        items: 3
      },
      1600: {
        items: 3
      },
    }
  });
});

//logo slider
$(window).on("load", function () {
  $('.clients-logo-table').owlCarousel({       
      nav: true,
      navText : ["<i class='fa-solid fa-arrow-left-long'></i>","<i class='fa-solid fa-arrow-right-long'></i>"],
      dots: false,
      autoplay: true,
      loop: true,   
      center: false,   
      margin: 30,
      autoplayTimeout: 2000,
      autoplayHoverPause: false,
      autoHeight: true,
      smartSpeed: 2000,   
     responsive: {
      0: {
        items: 2
      },
      520: {
        items: 3
      },
      768: {
        items: 4
      },
      1200: {
        items: 5
      },
      1400: {
        items: 6
      },
      1600: {
        items: 6
      },
    }
  });
});

//Work slider
$(window).on("load", function () {
  $('.work-slider').owlCarousel({
    nav: true,
    navText : ["<i class='fa-solid fa-arrow-left-long'></i>","<i class='fa-solid fa-arrow-right-long'></i>"],
    items:2,      
    dots: false,
    autoplay: true,
    loop: true,   
    center: true, 
    onInitialized: counter,
    onTranslated: counter,  
    margin: 30,
    autoplayTimeout: 4500,
    autoplayHoverPause: true,   
    smartSpeed: 1000,  
    startPosition: 0,   
    slideTransition: 'linear',   
    responsiveClass: true,     
    responsive: {
      0: {
        items: 1,
        stagePadding:0,
      },     
      1300: {
        items: 1,
        stagePadding: 200,
      },
      1500: {
        items: 2
      },
    }  
  });
});

//owl work counter
function counter(event) {
  if (!event.namespace) {
    return;}
  var slides = event.relatedTarget;
  $('#counter').html(slides.relative(slides.current()) + 1 + '/' + slides.items().length);
}
  
//Video magnificPopup
 $('.video-play').magnificPopup({
    type: 'iframe',
    mainClass: 'mfp-fade',
    removalDelay: 160,
 });
  
//Wow animation
new WOW().init();
  
//Portfolio filter
$( window ).on( "load", function() {
  $('.grid').isotope({
    itemSelector: '.grid-item',
  });  
// filter items on button click
  $('.filter-button-group').on( 'click', 'li', function() {
    var filterValue = $(this).attr('data-filter');
    $('.grid').isotope({ filter: filterValue });
    $('.filter-button-group li').removeClass('active');
    $(this).addClass('active');
  });
})
  
//odometer
$(window).on("scroll", function () {
  function winScrollPosition() {
    var scrollPos = $(window).scrollTop(),
      winHeight = $(window).height();
    var scrollPosition = Math.round(scrollPos + winHeight / 1.2);
    return scrollPosition;
  }
  $(".odometer").each(function () {
    var elemOffset = $(this).offset().top;
    if (elemOffset < winScrollPosition()) {
      $(this).html($(this).data("count-to"));
    }
  });
});


//background img
$("[data-background]").each(function () {
  $(this).css("background-image", "url(" + $(this).attr("data-background") + ")")
})


//Scroll to top
$.scrollUp({
  animation: 'fade',
  scrollImg: {
    active: true,
    type: 'background'
  }
});

//end of the code
})(jQuery);